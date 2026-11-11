/**
 * CryptaVision — Custom GStreamer Raw TCP Network Sink (C++)
 * ==========================================================
 * Plugin: cryptanetworksink
 * Class:  GstBaseSink subclass
 *
 * Implements low-latency edge-to-cloud telemetry transmission using raw POSIX
 * TCP sockets (<sys/socket.h>). Completely bypasses the Python interpreter
 * during network transmission for maximum throughput and deterministic I/O.
 *
 * Binary Packet Protocol:
 * ┌──────────────────┬──────────────────┬───────────────┬──────────────────────────┐
 * │ Magic Number     │ Payload Length   │ Payload Type  │ Payload Bytes            │
 * │ 4 bytes (uint32) │ 4 bytes (uint32) │ 1 byte (uint8)│ N bytes (raw data)       │
 * │ 0x43525950 'CRYP'│ Network (BE)     │ 0x01=JSON     │ JSON text or             │
 * │                  │                  │ 0x02=CKKS     │ Serialized TenSEAL CKKS  │
 * └──────────────────┴──────────────────┴───────────────┴──────────────────────────┘
 *
 * Non-Blocking & Fault Tolerance:
 * - Uses SO_SNDTIMEO and TCP_NODELAY (disabling Nagle's algorithm).
 * - Handles SIGPIPE suppression (SO_NOSIGPIPE on Darwin, MSG_NOSIGNAL on Linux).
 * - Automatic reconnection: Drops frames safely on network disconnect without
 *   stalling or deadlocking the upstream GStreamer video processing pipeline.
 */

#include <gst/gst.h>
#include <gst/base/gstbasesink.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#include <poll.h>
#include <errno.h>
#include <cstring>
#include <atomic>
#include <string>

G_BEGIN_DECLS

#define CRYPTA_TYPE_NETWORK_SINK            (crypta_network_sink_get_type())
#define CRYPTA_NETWORK_SINK(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), CRYPTA_TYPE_NETWORK_SINK, CryptaNetworkSink))
#define CRYPTA_NETWORK_SINK_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  CRYPTA_TYPE_NETWORK_SINK, CryptaNetworkSinkClass))
#define CRYPTA_IS_NETWORK_SINK(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), CRYPTA_TYPE_NETWORK_SINK))
#define CRYPTA_IS_NETWORK_SINK_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  CRYPTA_TYPE_NETWORK_SINK))

// 4-byte Magic Signature: 'CRYP' (0x43525950)
static const uint32_t CRYPTA_MAGIC_NUMBER = 0x43525950;

// Payload Type Identifiers
enum CryptaPayloadType : uint8_t {
    PAYLOAD_TYPE_JSON = 0x01,  // Stream A: Semantic JSON metadata
    PAYLOAD_TYPE_CKKS = 0x02   // Stream B: Encrypted TenSEAL CKKS vector
};

#pragma pack(push, 1)
struct CryptaPacketHeader {
    uint32_t magic;    // Network byte order
    uint32_t length;   // Network byte order
    uint8_t  type;     // 0x01 = JSON, 0x02 = CKKS
};
#pragma pack(pop)

typedef struct _CryptaNetworkSink {
    GstBaseSink parent_instance;

    // Element properties
    gchar*   host;
    gint     port;
    guint    timeout_ms;
    gboolean auto_reconnect;

    // Socket handle & metrics
    int sock_fd;
    std::atomic<bool> connected{false};
    std::atomic<uint64_t> packets_sent{0};
    std::atomic<uint64_t> bytes_sent{0};
    std::atomic<uint64_t> drop_count{0};
} CryptaNetworkSink;

typedef struct _CryptaNetworkSinkClass {
    GstBaseSinkClass parent_class;
} CryptaNetworkSinkClass;

GType crypta_network_sink_get_type(void);

G_END_DECLS

// ── Property Enumeration ───────────────────────────────────────────────────
enum {
    PROP_0,
    PROP_HOST,
    PROP_PORT,
    PROP_TIMEOUT_MS,
    PROP_AUTO_RECONNECT,
    PROP_LAST
};

// ── Pad Templates ──────────────────────────────────────────────────────────
// Accepts any raw buffer (JSON strings or binary CKKS bytes)
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS("ANY")
);

G_DEFINE_TYPE(CryptaNetworkSink, crypta_network_sink, GST_TYPE_BASE_SINK)

// ── Forward Declarations ───────────────────────────────────────────────────
static void     crypta_network_sink_set_property(GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec);
static void     crypta_network_sink_get_property(GObject *object, guint prop_id, GValue *value, GParamSpec *pspec);
static void     crypta_network_sink_finalize(GObject *object);
static gboolean crypta_network_sink_start(GstBaseSink *sink);
static gboolean crypta_network_sink_stop(GstBaseSink *sink);
static GstFlowReturn crypta_network_sink_render(GstBaseSink *sink, GstBuffer *buffer);
static bool     crypta_network_sink_connect(CryptaNetworkSink *self);
static void     crypta_network_sink_disconnect(CryptaNetworkSink *self);
static bool     crypta_network_sink_send_all(CryptaNetworkSink *self, const uint8_t *data, size_t total_len);

// ── Class Initialization ───────────────────────────────────────────────────
static void crypta_network_sink_class_init(CryptaNetworkSinkClass *klass) {
    GObjectClass     *gobject_class   = G_OBJECT_CLASS(klass);
    GstElementClass  *element_class   = GST_ELEMENT_CLASS(klass);
    GstBaseSinkClass *basesink_class  = GST_BASE_SINK_CLASS(klass);

    gobject_class->set_property = crypta_network_sink_set_property;
    gobject_class->get_property = crypta_network_sink_get_property;
    gobject_class->finalize     = crypta_network_sink_finalize;

    // Install Element Properties
    g_object_class_install_property(gobject_class, PROP_HOST,
        g_param_spec_string("host", "Host",
            "Target cloud server hostname or IP address",
            "127.0.0.1", (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_PORT,
        g_param_spec_int("port", "Port",
            "Target cloud server TCP port",
            1, 65535, 5555, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_TIMEOUT_MS,
        g_param_spec_uint("timeout-ms", "Timeout (ms)",
            "Socket send timeout in milliseconds",
            10, 30000, 1000, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_AUTO_RECONNECT,
        g_param_spec_boolean("auto-reconnect", "Auto Reconnect",
            "Attempt background socket reconnect on network disconnect",
            TRUE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    gst_element_class_add_static_pad_template(element_class, &sink_template);

    gst_element_class_set_static_metadata(element_class,
        "CryptaVision POSIX TCP Network Sink",
        "Sink/Network",
        "Pushes JSON metadata and CKKS ciphertexts to the cloud via raw POSIX TCP sockets",
        "CryptaVision Edge Systems Team <edge@cryptavision.io>");

    // GstBaseSink Virtual Methods
    basesink_class->start  = crypta_network_sink_start;
    basesink_class->stop   = crypta_network_sink_stop;
    basesink_class->render = crypta_network_sink_render;
}

// ── Instance Initialization ────────────────────────────────────────────────
static void crypta_network_sink_init(CryptaNetworkSink *self) {
    self->host           = g_strdup("127.0.0.1");
    self->port           = 5555;
    self->timeout_ms     = 1000;
    self->auto_reconnect = TRUE;
    self->sock_fd        = -1;
    self->connected.store(false);

    // Disable GStreamer clock synchronization by default for low latency telemetry
    gst_base_sink_set_sync(GST_BASE_SINK(self), FALSE);
}

// ── Finalize (Destructor) ──────────────────────────────────────────────────
static void crypta_network_sink_finalize(GObject *object) {
    CryptaNetworkSink *self = CRYPTA_NETWORK_SINK(object);
    crypta_network_sink_disconnect(self);
    g_free(self->host);
    G_OBJECT_CLASS(crypta_network_sink_parent_class)->finalize(object);
}

// ── Property Accessors ─────────────────────────────────────────────────────
static void crypta_network_sink_set_property(
    GObject *object, guint prop_id, const GValue *value, GParamSpec *pspec)
{
    CryptaNetworkSink *self = CRYPTA_NETWORK_SINK(object);
    switch (prop_id) {
        case PROP_HOST:
            g_free(self->host);
            self->host = g_value_dup_string(value);
            break;
        case PROP_PORT:
            self->port = g_value_get_int(value);
            break;
        case PROP_TIMEOUT_MS:
            self->timeout_ms = g_value_get_uint(value);
            break;
        case PROP_AUTO_RECONNECT:
            self->auto_reconnect = g_value_get_boolean(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    }
}

static void crypta_network_sink_get_property(
    GObject *object, guint prop_id, GValue *value, GParamSpec *pspec)
{
    CryptaNetworkSink *self = CRYPTA_NETWORK_SINK(object);
    switch (prop_id) {
        case PROP_HOST:
            g_value_set_string(value, self->host);
            break;
        case PROP_PORT:
            g_value_set_int(value, self->port);
            break;
        case PROP_TIMEOUT_MS:
            g_value_set_uint(value, self->timeout_ms);
            break;
        case PROP_AUTO_RECONNECT:
            g_value_set_boolean(value, self->auto_reconnect);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(object, prop_id, pspec);
    }
}

// ── Low-Level POSIX Socket Implementation ──────────────────────────────────
static bool crypta_network_sink_connect(CryptaNetworkSink *self) {
    if (self->connected.load() && self->sock_fd >= 0) {
        return true;
    }

    crypta_network_sink_disconnect(self);

    GST_INFO_OBJECT(self, "Connecting POSIX TCP socket to %s:%d...", self->host, self->port);

    struct addrinfo hints, *res = nullptr;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family   = AF_INET; // IPv4
    hints.ai_socktype = SOCK_STREAM;

    std::string port_str = std::to_string(self->port);
    int rc = getaddrinfo(self->host, port_str.c_str(), &hints, &res);
    if (rc != 0 || res == nullptr) {
        GST_WARNING_OBJECT(self, "getaddrinfo failed for %s:%d (%s)", self->host, self->port, gai_strerror(rc));
        return false;
    }

    int fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd < 0) {
        GST_WARNING_OBJECT(self, "socket creation failed: %s", strerror(errno));
        freeaddrinfo(res);
        return false;
    }

    // Set Send Timeout (SO_SNDTIMEO) to prevent blocking GStreamer streaming thread
    struct timeval tv;
    tv.tv_sec  = self->timeout_ms / 1000;
    tv.tv_usec = (self->timeout_ms % 1000) * 1000;
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (const char*)&tv, sizeof(tv));

    // Disable Nagle's algorithm for immediate frame telemetry dispatch
    int nodelay = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char*)&nodelay, sizeof(nodelay));

#ifdef SO_NOSIGPIPE
    // macOS: Prevent SIGPIPE on broken socket
    int nosigpipe = 1;
    setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &nosigpipe, sizeof(nosigpipe));
#endif

    // Attempt synchronous connect
    if (connect(fd, res->ai_addr, res->ai_addrlen) < 0) {
        GST_WARNING_OBJECT(self, "connect() to %s:%d failed: %s", self->host, self->port, strerror(errno));
        close(fd);
        freeaddrinfo(res);
        return false;
    }

    freeaddrinfo(res);
    self->sock_fd = fd;
    self->connected.store(true);
    GST_INFO_OBJECT(self, "Successfully connected POSIX TCP socket (fd=%d) to cloud server.", fd);
    return true;
}

static void crypta_network_sink_disconnect(CryptaNetworkSink *self) {
    if (self->sock_fd >= 0) {
        close(self->sock_fd);
        self->sock_fd = -1;
    }
    self->connected.store(false);
}

static bool crypta_network_sink_send_all(CryptaNetworkSink *self, const uint8_t *data, size_t total_len) {
    if (!self->connected.load() || self->sock_fd < 0) {
        if (self->auto_reconnect) {
            if (!crypta_network_sink_connect(self)) {
                return false;
            }
        } else {
            return false;
        }
    }

    size_t bytes_sent_total = 0;
    const int flags =
#ifdef MSG_NOSIGNAL
        MSG_NOSIGNAL; // Linux: suppress SIGPIPE
#else
        0;
#endif

    while (bytes_sent_total < total_len) {
        ssize_t n = send(self->sock_fd, data + bytes_sent_total, total_len - bytes_sent_total, flags);
        if (n > 0) {
            bytes_sent_total += n;
        } else if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // Timeout exceeded
                GST_WARNING_OBJECT(self, "send() timed out after %u ms", self->timeout_ms);
                return false;
            }
            // Broken pipe or connection reset
            GST_WARNING_OBJECT(self, "send() failed (%s); disconnecting socket", strerror(errno));
            crypta_network_sink_disconnect(self);
            return false;
        } else {
            // Socket closed cleanly by remote host
            GST_WARNING_OBJECT(self, "Remote host closed connection");
            crypta_network_sink_disconnect(self);
            return false;
        }
    }

    self->bytes_sent.fetch_add(total_len, std::memory_order_relaxed);
    self->packets_sent.fetch_add(1, std::memory_order_relaxed);
    return true;
}

// ── Lifecycle Start / Stop ─────────────────────────────────────────────────
static gboolean crypta_network_sink_start(GstBaseSink *sink) {
    CryptaNetworkSink *self = CRYPTA_NETWORK_SINK(sink);
    GST_INFO_OBJECT(self, "Starting cryptanetworksink element...");
    // Attempt initial connection (non-fatal if server is temporarily offline)
    crypta_network_sink_connect(self);
    return TRUE;
}

static gboolean crypta_network_sink_stop(GstBaseSink *sink) {
    CryptaNetworkSink *self = CRYPTA_NETWORK_SINK(sink);
    GST_INFO_OBJECT(self, "Stopping cryptanetworksink (packets=%lu, bytes=%lu, drops=%lu)...",
                    (unsigned long)self->packets_sent.load(),
                    (unsigned long)self->bytes_sent.load(),
                    (unsigned long)self->drop_count.load());
    crypta_network_sink_disconnect(self);
    return TRUE;
}

// ── Buffer Render (Hot Path) ───────────────────────────────────────────────
static GstFlowReturn crypta_network_sink_render(GstBaseSink *sink, GstBuffer *buffer) {
    CryptaNetworkSink *self = CRYPTA_NETWORK_SINK(sink);

    GstMapInfo map;
    if (!gst_buffer_map(buffer, &map, GST_MAP_READ)) {
        GST_ERROR_OBJECT(self, "Failed to map input buffer in render()");
        return GST_FLOW_ERROR;
    }

    if (map.size == 0) {
        gst_buffer_unmap(buffer, &map);
        return GST_FLOW_OK;
    }

    // Determine Payload Type:
    // 1. Explicit 1-byte prefix: 0x01 (JSON) or 0x02 (CKKS)
    // 2. Auto-detection: Starts with '{' or '[' -> JSON, otherwise CKKS binary
    uint8_t payload_type = PAYLOAD_TYPE_JSON;
    const uint8_t *payload_ptr = map.data;
    size_t payload_len = map.size;

    if (map.data[0] == PAYLOAD_TYPE_JSON || map.data[0] == PAYLOAD_TYPE_CKKS) {
        payload_type = map.data[0];
        payload_ptr  = map.data + 1;
        payload_len  = map.size - 1;
    } else if (map.data[0] == '{' || map.data[0] == '[') {
        payload_type = PAYLOAD_TYPE_JSON;
        payload_ptr  = map.data;
        payload_len  = map.size;
    } else {
        payload_type = PAYLOAD_TYPE_CKKS;
        payload_ptr  = map.data;
        payload_len  = map.size;
    }

    // Construct Binary Packet Protocol Header (9 bytes)
    CryptaPacketHeader header;
    header.magic  = htonl(CRYPTA_MAGIC_NUMBER);
    header.length = htonl(static_cast<uint32_t>(payload_len));
    header.type   = payload_type;

    // Send Header followed by Payload
    bool success = true;
    if (!crypta_network_sink_send_all(self, reinterpret_cast<const uint8_t*>(&header), sizeof(header))) {
        success = false;
    } else if (payload_len > 0) {
        if (!crypta_network_sink_send_all(self, payload_ptr, payload_len)) {
            success = false;
        }
    }

    if (!success) {
        self->drop_count.fetch_add(1, std::memory_order_relaxed);
        GST_DEBUG_OBJECT(self, "Telemetry packet dropped (total_drops=%lu)",
                         (unsigned long)self->drop_count.load());
    }

    gst_buffer_unmap(buffer, &map);

    // Always return GST_FLOW_OK so network hiccups never kill the GStreamer pipeline
    return GST_FLOW_OK;
}

// ── Plugin Registration Entrypoint ─────────────────────────────────────────
static gboolean plugin_init(GstPlugin *plugin) {
    return gst_element_register(
        plugin,
        "cryptanetworksink",
        GST_RANK_NONE,
        CRYPTA_TYPE_NETWORK_SINK
    );
}

#ifndef PACKAGE
#define PACKAGE "cryptavision"
#endif

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    crypta_network_sink,
    "CryptaVision Raw POSIX TCP Network Sink",
    plugin_init,
    "1.0.0",
    "Proprietary",
    "CryptaVision",
    "https://cryptavision.io"
)
