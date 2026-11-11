/**
 * CryptaVision — Custom GStreamer Base Transform Plugin (C++ Boilerplate)
 * =========================================================================
 * This file provides the structural skeleton for writing a custom GStreamer
 * plugin in C++ that can be used as a frame filter in either stream branch.
 *
 * Plugin type: GstBaseTransform (in-place capable)
 *   - Receives one input buffer, produces one output buffer.
 *   - Can be used in "in-place" mode (modify buffer directly, zero copy) or
 *     "copy" mode (allocate a new output buffer).
 *
 * To register this plugin:
 *   gst-inspect-1.0 crypta-frame-filter
 *
 * Build with CMake (see CMakeLists.txt in plugins/).
 *
 * References:
 *   https://gstreamer.freedesktop.org/documentation/plugin-development/
 *
 * Memory / thread model:
 *   - transform_ip() is called from the streaming thread.
 *   - Never hold a mutex across a Gst API call that may block.
 *   - GST_BUFFER_FLAG_READONLY: check before mapping WRITE — if set, request
 *     a copy (is handled automatically by GstBaseTransform in copy mode).
 */

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <gst/video/video.h>
#include <cstring>
#include <atomic>

G_BEGIN_DECLS

// ── Type macros ────────────────────────────────────────────────────────────
#define CRYPTA_TYPE_FRAME_FILTER            (crypta_frame_filter_get_type())
#define CRYPTA_FRAME_FILTER(obj)            (G_TYPE_CHECK_INSTANCE_CAST((obj), CRYPTA_TYPE_FRAME_FILTER, CryptaFrameFilter))
#define CRYPTA_FRAME_FILTER_CLASS(klass)    (G_TYPE_CHECK_CLASS_CAST((klass),  CRYPTA_TYPE_FRAME_FILTER, CryptaFrameFilterClass))
#define CRYPTA_IS_FRAME_FILTER(obj)         (G_TYPE_CHECK_INSTANCE_TYPE((obj), CRYPTA_TYPE_FRAME_FILTER))
#define CRYPTA_IS_FRAME_FILTER_CLASS(klass) (G_TYPE_CHECK_CLASS_TYPE((klass),  CRYPTA_TYPE_FRAME_FILTER))

// ── Instance structure ─────────────────────────────────────────────────────
typedef struct _CryptaFrameFilter {
    GstBaseTransform parent_instance;

    // Properties (set via g_object_set / element properties)
    gboolean    enabled;           // master on/off switch
    guint       obfuscation_mode;  // 0=passthrough, 1=blur, 2=pixelate, …

    // Runtime state
    std::atomic<guint64>  frame_counter{0};
    GstVideoInfo          video_info;     // parsed from caps negotiation
    gboolean              info_valid;
} CryptaFrameFilter;

typedef struct _CryptaFrameFilterClass {
    GstBaseTransformClass parent_class;
} CryptaFrameFilterClass;

GType crypta_frame_filter_get_type(void);

G_END_DECLS


// ── Property enum ──────────────────────────────────────────────────────────
enum {
    PROP_0,
    PROP_ENABLED,
    PROP_OBFUSCATION_MODE,
    PROP_LAST
};

// ── Pad templates ──────────────────────────────────────────────────────────
// Accept BGRx or I420 — match what the Python pipeline negotiates.
static GstStaticPadTemplate sink_template = GST_STATIC_PAD_TEMPLATE(
    "sink",
    GST_PAD_SINK,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        "video/x-raw, format=(string){ BGRx, I420 }, "
        "width=(int)[1,MAX], height=(int)[1,MAX], "
        "framerate=(fraction)[0/1,MAX]"
    )
);

static GstStaticPadTemplate src_template = GST_STATIC_PAD_TEMPLATE(
    "src",
    GST_PAD_SRC,
    GST_PAD_ALWAYS,
    GST_STATIC_CAPS(
        "video/x-raw, format=(string){ BGRx, I420 }, "
        "width=(int)[1,MAX], height=(int)[1,MAX], "
        "framerate=(fraction)[0/1,MAX]"
    )
);

// ── GObject boilerplate macro ──────────────────────────────────────────────
G_DEFINE_TYPE(CryptaFrameFilter, crypta_frame_filter, GST_TYPE_BASE_TRANSFORM)


// ── Forward declarations ───────────────────────────────────────────────────
static void     crypta_frame_filter_set_property(GObject*, guint, const GValue*, GParamSpec*);
static void     crypta_frame_filter_get_property(GObject*, guint, GValue*, GParamSpec*);
static void     crypta_frame_filter_finalize    (GObject*);
static gboolean crypta_frame_filter_set_caps    (GstBaseTransform*, GstCaps*, GstCaps*);
static GstFlowReturn crypta_frame_filter_transform_ip(GstBaseTransform*, GstBuffer*);
static gboolean crypta_frame_filter_start       (GstBaseTransform*);
static gboolean crypta_frame_filter_stop        (GstBaseTransform*);


// ── Class init ────────────────────────────────────────────────────────────
static void crypta_frame_filter_class_init(CryptaFrameFilterClass* klass) {
    GObjectClass*          gobject_class = G_OBJECT_CLASS(klass);
    GstElementClass*       element_class = GST_ELEMENT_CLASS(klass);
    GstBaseTransformClass* bt_class      = GST_BASE_TRANSFORM_CLASS(klass);

    // GObject virtual methods
    gobject_class->set_property = crypta_frame_filter_set_property;
    gobject_class->get_property = crypta_frame_filter_get_property;
    gobject_class->finalize     = crypta_frame_filter_finalize;

    // Install element properties (accessible from gst-launch-1.0 or Python)
    g_object_class_install_property(gobject_class, PROP_ENABLED,
        g_param_spec_boolean("enabled", "Enabled",
            "Enable or bypass the privacy filter",
            TRUE, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    g_object_class_install_property(gobject_class, PROP_OBFUSCATION_MODE,
        g_param_spec_uint("mode", "Obfuscation Mode",
            "0=passthrough, 1=blur, 2=pixelate",
            0, 10, 0, (GParamFlags)(G_PARAM_READWRITE | G_PARAM_STATIC_STRINGS)));

    // Pad templates
    gst_element_class_add_static_pad_template(element_class, &sink_template);
    gst_element_class_add_static_pad_template(element_class, &src_template);

    gst_element_class_set_static_metadata(element_class,
        "CryptaVision Frame Filter",
        "Filter/Video",
        "Privacy-preserving per-frame visual transform",
        "CryptaVision Core Team <dev@cryptavision.io>"
    );

    // BaseTransform virtual methods
    bt_class->set_caps      = crypta_frame_filter_set_caps;
    bt_class->transform_ip  = crypta_frame_filter_transform_ip;
    bt_class->start         = crypta_frame_filter_start;
    bt_class->stop          = crypta_frame_filter_stop;

    // Declare in-place capability:
    // When passthrough is not engaged, transform_ip modifies the buffer IN PLACE.
    // GstBaseTransform will automatically request a writable copy if the buffer
    // is read-only (e.g., shared with another branch via tee).
    bt_class->passthrough_on_same_caps = FALSE;
    GST_BASE_TRANSFORM_CLASS(klass)->transform_ip_on_passthrough = FALSE;
}

// ── Instance init ─────────────────────────────────────────────────────────
static void crypta_frame_filter_init(CryptaFrameFilter* self) {
    self->enabled          = TRUE;
    self->obfuscation_mode = 0;
    self->info_valid       = FALSE;

    // Operate in in-place mode by default (single buffer, no allocation)
    gst_base_transform_set_in_place(GST_BASE_TRANSFORM(self), TRUE);
    // Disable passthrough so transform_ip is always called
    gst_base_transform_set_passthrough(GST_BASE_TRANSFORM(self), FALSE);
}

// ── Finalize (destructor) ─────────────────────────────────────────────────
static void crypta_frame_filter_finalize(GObject* object) {
    // Release any heap allocations made in init here.
    // self->model_handle = nullptr; etc.
    G_OBJECT_CLASS(crypta_frame_filter_parent_class)->finalize(object);
}

// ── Properties ────────────────────────────────────────────────────────────
static void crypta_frame_filter_set_property(
    GObject* obj, guint prop_id, const GValue* value, GParamSpec* pspec)
{
    CryptaFrameFilter* self = CRYPTA_FRAME_FILTER(obj);
    switch (prop_id) {
        case PROP_ENABLED:
            self->enabled = g_value_get_boolean(value);
            GST_INFO_OBJECT(self, "enabled=%s", self->enabled ? "true" : "false");
            break;
        case PROP_OBFUSCATION_MODE:
            self->obfuscation_mode = g_value_get_uint(value);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec);
    }
}

static void crypta_frame_filter_get_property(
    GObject* obj, guint prop_id, GValue* value, GParamSpec* pspec)
{
    CryptaFrameFilter* self = CRYPTA_FRAME_FILTER(obj);
    switch (prop_id) {
        case PROP_ENABLED:
            g_value_set_boolean(value, self->enabled);
            break;
        case PROP_OBFUSCATION_MODE:
            g_value_set_uint(value, self->obfuscation_mode);
            break;
        default:
            G_OBJECT_WARN_INVALID_PROPERTY_ID(obj, prop_id, pspec);
    }
}

// ── Caps negotiation ──────────────────────────────────────────────────────
static gboolean crypta_frame_filter_set_caps(
    GstBaseTransform* bt, GstCaps* incaps, GstCaps* outcaps)
{
    CryptaFrameFilter* self = CRYPTA_FRAME_FILTER(bt);
    if (!gst_video_info_from_caps(&self->video_info, incaps)) {
        GST_ERROR_OBJECT(self, "Failed to parse input caps");
        return FALSE;
    }
    self->info_valid = TRUE;
    GST_INFO_OBJECT(self, "caps set: %dx%d %s",
        GST_VIDEO_INFO_WIDTH(&self->video_info),
        GST_VIDEO_INFO_HEIGHT(&self->video_info),
        gst_video_format_to_string(GST_VIDEO_INFO_FORMAT(&self->video_info))
    );
    return TRUE;
}

// ── Start / Stop ──────────────────────────────────────────────────────────
static gboolean crypta_frame_filter_start(GstBaseTransform* bt) {
    CryptaFrameFilter* self = CRYPTA_FRAME_FILTER(bt);
    self->frame_counter.store(0);
    GST_INFO_OBJECT(self, "Plugin started.");
    return TRUE;
}

static gboolean crypta_frame_filter_stop(GstBaseTransform* bt) {
    CryptaFrameFilter* self = CRYPTA_FRAME_FILTER(bt);
    GST_INFO_OBJECT(self, "Plugin stopped. Processed %lu frames.",
                    (unsigned long)self->frame_counter.load());
    return TRUE;
}

// ── Core processing ───────────────────────────────────────────────────────
/**
 * transform_ip — IN-PLACE frame processing.
 *
 * Memory contract:
 *   1. buf is guaranteed writable by GstBaseTransform before this is called
 *      (it will have made a writable copy if the original was read-only).
 *   2. Map the buffer for READ_WRITE.
 *   3. Operate directly on map.data — no extra allocation needed.
 *   4. Unmap in ALL code paths (use RAII or goto cleanup).
 *   5. Return GST_FLOW_OK to continue; GST_FLOW_ERROR to abort the pipeline.
 */
static GstFlowReturn crypta_frame_filter_transform_ip(
    GstBaseTransform* bt, GstBuffer* buf)
{
    CryptaFrameFilter* self = CRYPTA_FRAME_FILTER(bt);

    if (!self->enabled) {
        // Passthrough — return without touching the buffer
        return GST_FLOW_OK;
    }

    if (!self->info_valid) {
        GST_WARNING_OBJECT(self, "Video info not set; skipping frame.");
        return GST_FLOW_OK;
    }

    // ── Map buffer for in-place read/write ─────────────────────────────
    GstMapInfo map;
    if (!gst_buffer_map(buf, &map, GST_MAP_READWRITE)) {
        GST_ERROR_OBJECT(self, "gst_buffer_map failed");
        return GST_FLOW_ERROR;
    }

    const guint width  = GST_VIDEO_INFO_WIDTH (&self->video_info);
    const guint height = GST_VIDEO_INFO_HEIGHT(&self->video_info);
    const GstVideoFormat fmt = GST_VIDEO_INFO_FORMAT(&self->video_info);

    guint8* pixels = map.data;

    // ── Apply selected privacy transform ───────────────────────────────
    switch (self->obfuscation_mode) {
        case 0:
            // Passthrough — do nothing (enabled=true but mode=0 means audit only)
            break;

        case 1: {
            // Example: box blur 3x3 (in-place, BGRx)
            // Replace with SIMD-optimised or OpenCV call in production.
            if (fmt == GST_VIDEO_FORMAT_BGRx) {
                const guint stride = width * 4;   // 4 bytes per BGRx pixel
                for (guint y = 1; y < height - 1; ++y) {
                    for (guint x = 1; x < width - 1; ++x) {
                        for (guint c = 0; c < 3; ++c) {   // B, G, R (skip X)
                            guint32 sum = 0;
                            for (gint dy = -1; dy <= 1; ++dy)
                                for (gint dx = -1; dx <= 1; ++dx)
                                    sum += pixels[(y+dy)*stride + (x+dx)*4 + c];
                            pixels[y*stride + x*4 + c] = (guint8)(sum / 9);
                        }
                    }
                }
            }
            break;
        }

        case 2: {
            // Example: 16x16 pixelation (BGRx)
            if (fmt == GST_VIDEO_FORMAT_BGRx) {
                const guint stride  = width * 4;
                const guint block   = 16;
                for (guint by = 0; by < height; by += block) {
                    for (guint bx = 0; bx < width; bx += block) {
                        // compute block mean
                        guint32 sum[3] = {0, 0, 0};
                        guint   count  = 0;
                        for (guint y = by; y < MIN(by+block, height); ++y) {
                            for (guint x = bx; x < MIN(bx+block, width); ++x) {
                                for (guint c = 0; c < 3; ++c)
                                    sum[c] += pixels[y*stride + x*4 + c];
                                ++count;
                            }
                        }
                        guint8 mean[3] = {
                            (guint8)(sum[0]/count),
                            (guint8)(sum[1]/count),
                            (guint8)(sum[2]/count)
                        };
                        // fill block with mean
                        for (guint y = by; y < MIN(by+block, height); ++y)
                            for (guint x = bx; x < MIN(bx+block, width); ++x)
                                for (guint c = 0; c < 3; ++c)
                                    pixels[y*stride + x*4 + c] = mean[c];
                    }
                }
            }
            break;
        }

        default:
            GST_WARNING_OBJECT(self, "Unknown obfuscation mode: %u", self->obfuscation_mode);
    }

    // ── Mandatory unmap before returning ──────────────────────────────
    gst_buffer_unmap(buf, &map);

    self->frame_counter.fetch_add(1, std::memory_order_relaxed);
    return GST_FLOW_OK;
}


// ══════════════════════════════════════════════════════════════════════════
//  Plugin registration entry-point
// ══════════════════════════════════════════════════════════════════════════
static gboolean plugin_init(GstPlugin* plugin) {
    return gst_element_register(
        plugin,
        "crypta-frame-filter",        // element factory name
        GST_RANK_NONE,                // priority (NONE = don't use auto-plug)
        CRYPTA_TYPE_FRAME_FILTER
    );
}

// Metadata that gst-inspect-1.0 and the plugin scanner read
#ifndef PACKAGE
#define PACKAGE "cryptavision"
#endif

GST_PLUGIN_DEFINE(
    GST_VERSION_MAJOR,
    GST_VERSION_MINOR,
    crypta_frame_filter,
    "CryptaVision privacy frame filter plugin",
    plugin_init,
    "1.0.0",
    "Proprietary",
    "CryptaVision",
    "https://cryptavision.io"
)
