#include "texture.h"

#include <GL/gl.h>
#include "../glx/hardext.h"
#include "../glx/streaming.h"
#include "GL/glext.h"
#include "array.h"
#include "blit.h"
#include "decompress.h"
#include "debug.h"
#include "enum_info.h"
#include "fpe.h"
#include "framebuffers.h"
#include "gles.h"
#include "init.h"
#include "loader.h"
#include "matrix.h"
#include "pixel.h"
#include "raster.h"

// #define DEBUG
#ifdef DEBUG
#define DBG(a) a
#define DBGLOGD(...) SHUT_LOGD(__VA_ARGS__)
#else
#define DBG(a)
#define DBGLOGD(...)                                                                                                   \
    {}
#endif

#ifndef GL_TEXTURE_STREAM_IMG
#define GL_TEXTURE_STREAM_IMG 0x8C0D
#endif
#ifdef TEXSTREAM
#include <EGL/egl.h>
#include <EGL/eglext.h>
#endif

// expand non-power-of-two sizes
// TODO: what does this do to repeating textures?
int npot(int n) {
    if (n == 0) return 0;

    int i = 1;
    while (i < n)
        i <<= 1;
    return i;
}

static int inline nlevel(int size, int level) {
    if (size) {
        size >>= level;
        if (!size) size = 1;
    }
    return size;
}

// return the max level for that WxH size
static int inline maxlevel(int w, int h) {
    int mlevel = 0;
    while (w != 1 || h != 1) {
        w >>= 1;
        h >>= 1;
        if (!w) w = 1;
        if (!h) h = 1;
        ++mlevel;
    }
    return mlevel;
}

static inline GLboolean bgra_supported_type(GLenum type) {
    // GL_EXT_texture_format_BGRA8888 only guarantees BGRA with UNSIGNED_BYTE on GLES.
    return hardext.bgra8888 && (type == GL_UNSIGNED_BYTE);
}

static int is_fake_compressed_rgb(GLenum internalformat) {
    if (internalformat == GL_COMPRESSED_RGB) return 1;
    if (internalformat == GL_COMPRESSED_RGB_S3TC_DXT1_EXT) return 1;
    if (internalformat == GL_COMPRESSED_SRGB_S3TC_DXT1_EXT) return 1;
    return 0;
}
static int is_fake_compressed_rgba(GLenum internalformat) {
    if (internalformat == GL_COMPRESSED_RGBA) return 1;
    if (internalformat == GL_COMPRESSED_RGBA_S3TC_DXT1_EXT) return 1;
    if (internalformat == GL_COMPRESSED_RGBA_S3TC_DXT3_EXT) return 1;
    if (internalformat == GL_COMPRESSED_RGBA_S3TC_DXT5_EXT) return 1;
    if (internalformat == GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT1_EXT) return 1;
    if (internalformat == GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT3_EXT) return 1;
    if (internalformat == GL_COMPRESSED_SRGB_ALPHA_S3TC_DXT5_EXT) return 1;
    return 0;
}

// The real function to convert format
void internal_convert(GLenum* internal_format, GLenum* type, GLenum* format) {
    if (format && (*format == GL_BGRA || *format == GL_BGR || *format == GL_BGRA8_EXT)) return;
    if (type && *type == GL_UNSIGNED_INT_8_8_8_8) return;

    switch (*internal_format) {
    case GL_DEPTH_COMPONENT16:
        if (type) *type = GL_UNSIGNED_SHORT;
        break;
    case GL_DEPTH_COMPONENT24:
        if (type) *type = GL_UNSIGNED_INT;
        break;
    case GL_DEPTH_COMPONENT32:
        *internal_format = GL_DEPTH_COMPONENT;
        if (type) *type = GL_UNSIGNED_INT;
        break;
    case GL_DEPTH_COMPONENT32F:
        if (type) *type = GL_FLOAT;
        break;
    case GL_DEPTH_COMPONENT:
        if (type) {
            *internal_format = GL_DEPTH_COMPONENT;
            *type = GL_UNSIGNED_INT;
        }
        break;
    case GL_DEPTH_STENCIL:
        *internal_format = GL_DEPTH32F_STENCIL8;
        if (type) *type = GL_FLOAT_32_UNSIGNED_INT_24_8_REV;
        break;
    case GL_RGB10_A2:
        if (type) *type = GL_UNSIGNED_INT_2_10_10_10_REV;
        break;
    case GL_RGB5_A1:
        if (type) *type = GL_UNSIGNED_SHORT_5_5_5_1;
        break;
    case GL_COMPRESSED_RED_RGTC1:
    case GL_COMPRESSED_RG_RGTC2:
        break;
    case GL_SRGB8:
        if (type) *type = GL_UNSIGNED_BYTE;
        break;
    case GL_RGBA32F:
    case GL_RGB32F:
        if (type) *type = GL_FLOAT;
        break;
    case GL_RGB9_E5:
        if (type) *type = GL_UNSIGNED_INT_5_9_9_9_REV;
        break;
    case GL_R11F_G11F_B10F:
        if (type) *type = GL_UNSIGNED_INT_10F_11F_11F_REV;
        if (format) *format = GL_RGB;
        break;
    case GL_RGBA32UI:
    case GL_RGB32UI:
        if (type) *type = GL_UNSIGNED_INT;
        break;
    case GL_RGBA32I:
    case GL_RGB32I:
        if (type) *type = GL_INT;
        break;
    case GL_RGBA16: {
        *internal_format = GL_RGBA16F;
        if (type) *type = GL_FLOAT;
        break;
    }
    case GL_RGBA8:
    case GL_RGBA:
        if (type) *type = GL_UNSIGNED_BYTE;
        if (format) *format = GL_RGBA;
        break;
    case GL_RGBA16F:
        if (type) *type = GL_HALF_FLOAT;
        break;
    case GL_R16:
        *internal_format = GL_R16F;
        if (type) *type = GL_FLOAT;
        break;
    case GL_RGB16:
        *internal_format = GL_RGB16F;
        if (type) *type = GL_HALF_FLOAT;
        if (format) *format = GL_RGB;
        break;
    case GL_RGB16F:
        if (type) *type = GL_HALF_FLOAT;
        if (format) *format = GL_RGB;
        break;
    case GL_RG16:
        *internal_format = GL_RG16F;
        if (type) *type = GL_HALF_FLOAT;
        if (format) *format = GL_RG;
        break;
        // Inline R and RG channel mappings
    case GL_R8:
        if (format) *format = GL_RED;
        if (type) *type = GL_UNSIGNED_BYTE;
        break;
    case GL_R8_SNORM:
        if (format) *format = GL_RED;
        if (type) *type = GL_BYTE;
        break;
    case GL_R16F:
        if (format) *format = GL_RED;
        if (type) *type = GL_HALF_FLOAT;
        break;
    case GL_RED:
        if (type) {
            switch (*type) {
            case GL_UNSIGNED_BYTE:
                *internal_format = GL_R8;
                if (format) *format = GL_RED;
                break;
            case GL_BYTE:
                *internal_format = GL_R8_SNORM;
                if (format) *format = GL_RED;
                break;
            case GL_HALF_FLOAT:
                *internal_format = GL_R16F;
                if (format) *format = GL_RED;
                break;
            case GL_FLOAT:
                *internal_format = GL_R32F;
                if (format) *format = GL_RED;
                break;
            default:
                if (type) *type = GL_UNSIGNED_BYTE; // Fallback to unsigned byte
                *internal_format = GL_R8;           // Fallback to R8
                if (format) *format = GL_RED;
                break;
            }
        }
        break;
    case GL_R8UI:
        if (format) *format = GL_RED_INTEGER;
        if (type) *type = GL_UNSIGNED_BYTE;
        break;
    case GL_R8I:
        if (format) *format = GL_RED_INTEGER;
        if (type) *type = GL_BYTE;
        break;
    case GL_R16UI:
        if (format) *format = GL_RED_INTEGER;
        if (type) *type = GL_UNSIGNED_SHORT;
        break;
    case GL_R16I:
        if (format) *format = GL_RED_INTEGER;
        if (type) *type = GL_SHORT;
        break;
    case GL_R32UI:
        if (format) *format = GL_RED_INTEGER;
        if (type) *type = GL_UNSIGNED_INT;
        break;
    case GL_R32I:
        if (format) *format = GL_RED_INTEGER;
        if (type) *type = GL_INT;
        break;
    case GL_RG8:
        if (format) *format = GL_RG;
        if (type) *type = GL_UNSIGNED_BYTE;
        break;
    case GL_RG8_SNORM:
        if (format) *format = GL_RG;
        if (type) *type = GL_BYTE;
        break;
    case GL_RG16F:
        if (format) *format = GL_RG;
        if (type) *type = GL_HALF_FLOAT;
        break;
    case GL_RG32F:
        if (format) *format = GL_RG;
        if (type) *type = GL_FLOAT;
        break;
    case GL_RG8UI:
        if (format) *format = GL_RG_INTEGER;
        if (type) *type = GL_UNSIGNED_BYTE;
        break;
    case GL_RG8I:
        if (format) *format = GL_RG_INTEGER;
        if (type) *type = GL_BYTE;
        break;
    case GL_RG16UI:
        if (format) *format = GL_RG_INTEGER;
        if (type) *type = GL_UNSIGNED_SHORT;
        break;
    case GL_RG16I:
        if (format) *format = GL_RG_INTEGER;
        if (type) *type = GL_SHORT;
        break;
    case GL_RG32UI:
        if (format) *format = GL_RG_INTEGER;
        if (type) *type = GL_UNSIGNED_INT;
        break;
    case GL_RG32I:
        if (format) *format = GL_RG_INTEGER;
        if (type) *type = GL_INT;
        break;
    case GL_RGBA8_SNORM:
        if (format) *format = GL_RGBA;
        if (type) *type = GL_BYTE;
        break;
    case GL_R32F:
        if (format) *format = GL_RED;
        if (type) *type = GL_FLOAT;
        break;
    default:
        // fallback handling for GL_RGB8, GL_RGBA16_SNORM etc.
        if (*internal_format == GL_RGB8) {
            if (type && *type != GL_UNSIGNED_BYTE) *type = GL_UNSIGNED_BYTE;
            if (format) *format = GL_RGB;
        } else if (*internal_format == GL_RGBA16_SNORM) {
            if (type && *type != GL_SHORT) *type = GL_SHORT;
        }
        break;
    }
}

void internal2format_type(GLenum* internalformat, GLenum* format, GLenum* type) {
    if (format && *format != GL_BGRA && *format != GL_BGR && *format != GL_BGRA8_EXT) return;
    DBG(char log_buffer[512]; int offset = snprintf(log_buffer, sizeof(log_buffer), "tex format converting... ");
        if (internalformat) offset +=
        snprintf(log_buffer + offset, sizeof(log_buffer) - offset, "internalFormat: %s", PrintEnum(*internalformat));
        if (format) offset +=
        snprintf(log_buffer + offset, sizeof(log_buffer) - offset, ", format: %s", PrintEnum(*format));
        if (type) offset += snprintf(log_buffer + offset, sizeof(log_buffer) - offset, ", type: %s", PrintEnum(*type));
        snprintf(log_buffer + offset, sizeof(log_buffer) - offset, "\n"); DBGLOGD("%s", log_buffer))
    switch (*internalformat) {

    case GL_RGB10_A2:
        if (type) *type = GL_UNSIGNED_INT_2_10_10_10_REV;
        break;
    case GL_RGB5_A1:
        if (type) *type = GL_UNSIGNED_SHORT_5_5_5_1;
        break;
    case GL_SRGB8:
        if (type) *type = GL_UNSIGNED_BYTE;
        break;
    case GL_RGBA32F:
    case GL_RGB32F:
        if (type) *type = GL_FLOAT;
        break;
    case GL_RGB9_E5:
        if (type) *type = GL_UNSIGNED_INT_5_9_9_9_REV;
        break;
    case GL_R11F_G11F_B10F:
        if (type) *type = GL_UNSIGNED_INT_10F_11F_11F_REV;
        if (format) *format = GL_RGB;
        break;
    case GL_RGBA32UI:
    case GL_RGB32UI:
        if (type) *type = GL_UNSIGNED_INT;
        break;
    case GL_RGBA32I:
    case GL_RGB32I:
        if (type) *type = GL_INT;
        break;
    case GL_RGBA16: {
        *internalformat = GL_RGBA16F;
        if (type) *type = GL_FLOAT;
        break;
    }
    case GL_RGBA8:
    case GL_RGBA:
        if (type) *type = GL_UNSIGNED_BYTE;
        if (format) *format = GL_RGBA;
        break;
    case GL_RGBA16F:
        if (type) *type = GL_HALF_FLOAT;
        break;
    case GL_R16:
        *internalformat = GL_R16F;
        if (type) *type = GL_FLOAT;
        break;
    case GL_RGB16:
        *internalformat = GL_RGB16F;
        if (type) *type = GL_HALF_FLOAT;
        if (format) *format = GL_RGB;
        break;
    case GL_RGB16F:
        if (type) *type = GL_HALF_FLOAT;
        if (format) *format = GL_RGB;
        break;
    case GL_RG16:
        *internalformat = GL_RG16F;
        if (type) *type = GL_HALF_FLOAT;
        if (format) *format = GL_RG;
        break;
        // Inline R and RG channel mappings
    case GL_R8:
        if (format) *format = GL_RED;
        if (type) *type = GL_UNSIGNED_BYTE;
        break;
    case GL_R8_SNORM:
        if (format) *format = GL_RED;
        if (type) *type = GL_BYTE;
        break;
    case GL_R16F:
        if (format) *format = GL_RED;
        if (type) *type = GL_HALF_FLOAT;
        break;
    case GL_RED:
        if (type) {
            switch (*type) {
            case GL_UNSIGNED_BYTE:
                *internalformat = GL_R8;
                if (format) *format = GL_RED;
                break;
            case GL_BYTE:
                *internalformat = GL_R8_SNORM;
                if (format) *format = GL_RED;
                break;
            case GL_HALF_FLOAT:
                *internalformat = GL_R16F;
                if (format) *format = GL_RED;
                break;
            case GL_FLOAT:
                *internalformat = GL_R32F;
                if (format) *format = GL_RED;
                break;
            default:
                if (type) *type = GL_UNSIGNED_BYTE; // Fallback to unsigned byte
                *internalformat = GL_R8;            // Fallback to R8
                if (format) *format = GL_RED;
                break;
            }
        }
        break;
    case GL_R8UI:
        if (format) *format = GL_RED_INTEGER;
        if (type) *type = GL_UNSIGNED_BYTE;
        break;
    case GL_R8I:
        if (format) *format = GL_RED_INTEGER;
        if (type) *type = GL_BYTE;
        break;
    case GL_R16UI:
        if (format) *format = GL_RED_INTEGER;
        if (type) *type = GL_UNSIGNED_SHORT;
        break;
    case GL_R16I:
        if (format) *format = GL_RED_INTEGER;
        if (type) *type = GL_SHORT;
        break;
    case GL_R32UI:
        if (format) *format = GL_RED_INTEGER;
        if (type) *type = GL_UNSIGNED_INT;
        break;
    case GL_R32I:
        if (format) *format = GL_RED_INTEGER;
        if (type) *type = GL_INT;
        break;
    case GL_RG8:
        if (format) *format = GL_RG;
        if (type) *type = GL_UNSIGNED_BYTE;
        break;
    case GL_RG8_SNORM:
        if (format) *format = GL_RG;
        if (type) *type = GL_BYTE;
        break;
    case GL_RG16F:
        if (format) *format = GL_RG;
        if (type) *type = GL_HALF_FLOAT;
        break;
    case GL_RG32F:
        if (format) *format = GL_RG;
        if (type) *type = GL_FLOAT;
        break;
    case GL_RG8UI:
        if (format) *format = GL_RG_INTEGER;
        if (type) *type = GL_UNSIGNED_BYTE;
        break;
    case GL_RG8I:
        if (format) *format = GL_RG_INTEGER;
        if (type) *type = GL_BYTE;
        break;
    case GL_RG16UI:
        if (format) *format = GL_RG_INTEGER;
        if (type) *type = GL_UNSIGNED_SHORT;
        break;
    case GL_RG16I:
        if (format) *format = GL_RG_INTEGER;
        if (type) *type = GL_SHORT;
        break;
    case GL_RG32UI:
        if (format) *format = GL_RG_INTEGER;
        if (type) *type = GL_UNSIGNED_INT;
        break;
    case GL_RG32I:
        if (format) *format = GL_RG_INTEGER;
        if (type) *type = GL_INT;
        break;
    case GL_R:
        if (format) *format = GL_RED;
        if (type) *type = GL_UNSIGNED_BYTE;
        break;
    case GL_RG:
        if (!hardext.rgtex) {
            *format = GL_RGB;
            *type = GL_UNSIGNED_BYTE;
        } else {
            *format = GL_RG;
            *type = GL_UNSIGNED_BYTE;
        }
        break;
    case GL_COMPRESSED_ALPHA:
    case GL_ALPHA:
        *format = GL_ALPHA;
        *type = GL_UNSIGNED_BYTE;
        break;
    case 1: // is this here or with GL_RED?
    case GL_COMPRESSED_LUMINANCE:
    case GL_LUMINANCE:
        *format = GL_LUMINANCE;
        *type = GL_UNSIGNED_BYTE;
        break;
    case 2:
    case GL_COMPRESSED_LUMINANCE_ALPHA:
    case GL_LUMINANCE8_ALPHA8:
    case GL_LUMINANCE_ALPHA:
        if (globals4es.nolumalpha) {
            *format = GL_RGBA;
            *type = GL_UNSIGNED_BYTE;
        } else {
            *format = GL_LUMINANCE_ALPHA;
            *type = GL_UNSIGNED_BYTE;
        }
        break;
    case GL_RGB5:
    case GL_RGB565:
        *format = GL_RGB;
        *type = GL_UNSIGNED_SHORT_5_6_5;
        break;
    case GL_RGB:
        if (globals4es.avoid24bits)
            *format = GL_RGBA;
        else
            *format = GL_RGB;
        *type = GL_UNSIGNED_BYTE;
        break;
    case GL_RGBA4:
        *format = GL_RGBA;
        *type = GL_UNSIGNED_SHORT_4_4_4_4;
        break;
    case GL_BGRA:
        if (hardext.bgra8888)
            *format = GL_BGRA;
        else
            *format = GL_RGBA;
        *type = GL_UNSIGNED_BYTE;
        break;
    case GL_DEPTH_COMPONENT16:
        if (type) *type = GL_UNSIGNED_SHORT;
        break;

    case GL_DEPTH_COMPONENT24:
        if (type) *type = GL_UNSIGNED_INT;
        break;

    case GL_DEPTH_COMPONENT32:
        *internalformat = GL_DEPTH_COMPONENT;
        if (type) *type = GL_UNSIGNED_INT;
        break;

    case GL_DEPTH_COMPONENT32F:
        if (type) *type = GL_UNSIGNED_INT;
        break;
    case GL_DEPTH_COMPONENT:
        *format = GL_DEPTH_COMPONENT;
        *type = GL_UNSIGNED_INT;
        break;
    case GL_DEPTH_STENCIL:
    case GL_DEPTH24_STENCIL8:
        *format = GL_DEPTH_STENCIL;
        *type = GL_UNSIGNED_INT_24_8;
        break;
    default:
        // fallback handling for GL_RGB8, GL_RGBA16_SNORM etc.
        if (*internalformat == GL_RGB8) {
            if (type && *type != GL_UNSIGNED_BYTE) *type = GL_UNSIGNED_BYTE;
            if (format) *format = GL_RGB;
        } else if (*internalformat == GL_RGBA16_SNORM) {
            if (type && *type != GL_SHORT) *type = GL_SHORT;
        }
        break;
    }
    DBG(char log_buffer2[512]; int offset2 = snprintf(log_buffer, sizeof(log_buffer), "converted: ");
        if (internalformat) offset2 +=
        snprintf(log_buffer + offset2, sizeof(log_buffer) - offset2, "internalFormat: %s", PrintEnum(*internalformat));
        if (format) offset2 +=
        snprintf(log_buffer + offset2, sizeof(log_buffer) - offset2, ", format: %s", PrintEnum(*format));
        if (type) offset2 +=
        snprintf(log_buffer + offset2, sizeof(log_buffer) - offset2, ", type: %s", PrintEnum(*type));
        snprintf(log_buffer2 + offset2, sizeof(log_buffer2) - offset2, "\n"); DBGLOGD("%s", log_buffer))
}

static void* swizzle_texture(GLsizei width, GLsizei height, GLenum* format, GLenum* type, GLenum intermediaryformat,
                             GLenum internalformat, const GLvoid* data, gltexture_t* bound) {
    if (format && *format != GL_BGRA && *format != GL_BGR && *format != GL_BGRA8_EXT &&
        *type != GL_UNSIGNED_INT_8_8_8_8)
        return data;
    if (format && *format == GL_BGRA8_EXT) *format = GL_BGRA;
    int convert = 0;
    GLenum dest_format = GL_RGBA;
    GLenum dest_type = GL_UNSIGNED_BYTE;
    int check = 1;
    const GLboolean bgra_ok = bgra_supported_type(*type);
    // compressed format are not handled here, so mask them....
    if (is_fake_compressed_rgb(intermediaryformat)) intermediaryformat = GL_RGB;
    if (is_fake_compressed_rgba(intermediaryformat)) intermediaryformat = GL_RGBA;
    if (is_fake_compressed_rgb(internalformat)) internalformat = GL_RGB;
    if (is_fake_compressed_rgba(internalformat)) internalformat = GL_RGBA;
    if (intermediaryformat == GL_COMPRESSED_LUMINANCE) intermediaryformat = GL_LUMINANCE;
    if (internalformat == GL_COMPRESSED_LUMINANCE) internalformat = GL_LUMINANCE;

    // if (*format != intermediaryformat || intermediaryformat != internalformat) {
    //     internal2format_type(&intermediaryformat, &dest_format, &dest_type);
    //     convert = 1;
    //     check = 0;
    // } else

    {
        if ((*type) == GL_HALF_FLOAT) (*type) = GL_HALF_FLOAT_OES; // the define is different between GL and GLES...
        switch (*format) {
        case GL_R:
        case GL_RED:
            dest_format = GL_RED;
            check = 0;
            break;
        case GL_RG:
            dest_format = GL_RG;
            check = 0;
            break;
        case GL_COMPRESSED_LUMINANCE:
            *format = GL_LUMINANCE;
        case GL_LUMINANCE:
            dest_format = GL_LUMINANCE;
            break;
        case GL_LUMINANCE16F:
            dest_format = GL_LUMINANCE;
            if (hardext.halffloattex) {
                dest_type = GL_HALF_FLOAT_OES;
                check = 0;
            }
            break;
        case GL_LUMINANCE32F:
            dest_format = GL_LUMINANCE;
            if (hardext.floattex) {
                dest_type = GL_FLOAT;
                check = 0;
            }
            break;
        case GL_RGB:
            dest_format = GL_RGB;
            check = 0;
            break;
        case GL_COMPRESSED_ALPHA:
            *format = GL_ALPHA;
        case GL_ALPHA:
            dest_format = GL_ALPHA;
            break;
        case GL_ALPHA16F:
            dest_format = GL_ALPHA;
            if (hardext.halffloattex) {
                dest_type = GL_HALF_FLOAT_OES;
                check = 0;
            }
            break;
        case GL_ALPHA32F:
            dest_format = GL_ALPHA;
            if (hardext.floattex) {
                dest_type = GL_FLOAT;
                check = 0;
            }
            break;
        case GL_RGBA:
            check = 0;
            break;
        case GL_LUMINANCE8_ALPHA8:
        case GL_COMPRESSED_LUMINANCE_ALPHA:
            if (globals4es.nolumalpha)
                convert = 1;
            else {
                dest_format = GL_LUMINANCE_ALPHA;
                *format = GL_LUMINANCE_ALPHA;
            }
            break;
        case GL_LUMINANCE_ALPHA:
            if (globals4es.nolumalpha)
                convert = 1;
            else
                dest_format = GL_LUMINANCE_ALPHA;
            break;
        case GL_LUMINANCE_ALPHA16F:
            if (globals4es.nolumalpha)
                convert = 1;
            else
                dest_format = GL_LUMINANCE_ALPHA;
            if (hardext.halffloattex) {
                dest_type = GL_HALF_FLOAT_OES;
                check = 0;
            }
            break;
        case GL_LUMINANCE_ALPHA32F:
            if (globals4es.nolumalpha)
                convert = 1;
            else
                dest_format = GL_LUMINANCE_ALPHA;
            if (hardext.floattex) {
                dest_type = GL_FLOAT;
                check = 0;
            }
            break;
            // vvvvv all this are internal formats, so it should not happens
        case GL_RGB565:
            check = 0;
            break;
        case GL_RGB5:
            dest_format = GL_RGB;
            dest_type = GL_UNSIGNED_SHORT_5_6_5;
            convert = 1;
            check = 0;
            break;
        case GL_RGB8:
            check = 0;
            break;
        case GL_RGBA4:
            check = 0;
            break;
        case GL_RGBA8:
            check = 0;
            break;
        case GL_BGRA:
            if (bgra_ok) {
                dest_format = GL_BGRA;
                //*format = GL_BGRA;
            } else {
                convert = 1;
                dest_format = GL_RGBA;
            }
            break;
        case GL_BGR:
            dest_format = GL_RGB;
            convert = 1;
            break;
        case GL_DEPTH32F_STENCIL8:
        case GL_DEPTH24_STENCIL8:
        case GL_DEPTH_STENCIL: {
            // if (hardext.depthtex && hardext.depthstencil) {
            const int is32F = *format == GL_DEPTH32F_STENCIL8;
            *format = dest_format = GL_DEPTH_STENCIL;
            dest_type = is32F ? GL_FLOAT_32_UNSIGNED_INT_24_8_REV : GL_UNSIGNED_INT_24_8;
            //   check = 0;
            //}
            // else convert = 1;
            break;
        }
        case GL_DEPTH_COMPONENT:
            check = 0;
            // if (hardext.depthtex) {
            *format = dest_format = GL_DEPTH_COMPONENT;
            // if (dest_type != GL_UNSIGNED_INT) {
            //     convert = 1;
            // }
            dest_type = GL_UNSIGNED_INT;
            //    check = 0;
            //}
            // else
            //    convert = 1;
            break;
        case GL_DEPTH_COMPONENT16:
            check = 0;
            *format = dest_format = GL_DEPTH_COMPONENT;
            dest_type = GL_UNSIGNED_SHORT;
            break;
        case GL_DEPTH_COMPONENT24:
            check = 0;
            *format = dest_format = GL_DEPTH_COMPONENT;
            dest_type = GL_UNSIGNED_INT;
            break;
        case GL_DEPTH_COMPONENT32:
            check = 0;
            *format = dest_format = GL_DEPTH_COMPONENT;
            dest_type = GL_UNSIGNED_INT;
            break;
        case GL_DEPTH_COMPONENT32F:
            check = 0;
            *format = dest_format = GL_DEPTH_COMPONENT;
            dest_type = GL_FLOAT;
            break;
        case GL_STENCIL_INDEX8:
            check = 0;
            if (hardext.stenciltex)
                *format = dest_format = GL_STENCIL_INDEX8;
            else
                convert = 1;
            break;
        default:
            check = 0;
            // convert = 1;
            break;
        }
        if (check) switch (*type) {
            case GL_UNSIGNED_SHORT_4_4_4_4_REV:
                if (dest_format == GL_RGBA) dest_type = GL_UNSIGNED_SHORT_4_4_4_4;
                convert = 1;
                break;
            case GL_UNSIGNED_SHORT_4_4_4_4:
                if (dest_format == GL_RGBA)
                    dest_type = GL_UNSIGNED_SHORT_4_4_4_4;
                else
                    convert = 1;
                break;
            case GL_UNSIGNED_SHORT_1_5_5_5_REV:
                if (!hardext.rgba1555rev) {
                    if (dest_format == GL_RGBA) dest_type = GL_UNSIGNED_SHORT_5_5_5_1;
                    convert = 1;
                }
                break;
            case GL_UNSIGNED_SHORT_5_5_5_1:
                if (dest_format == GL_RGBA)
                    dest_type = GL_UNSIGNED_SHORT_5_5_5_1;
                else
                    convert = 1;
                break;
            case GL_UNSIGNED_SHORT_5_6_5_REV:
                if (dest_format == GL_RGB) dest_type = GL_UNSIGNED_SHORT_5_6_5;
                convert = 1;
                break;
            case GL_UNSIGNED_SHORT_5_6_5:
                if (dest_format == GL_RGB)
                    dest_type = GL_UNSIGNED_SHORT_5_6_5;
                else
                    convert = 1;
                break;
#ifdef __BIG_ENDIAN__
            case GL_UNSIGNED_INT_8_8_8_8:
#else
            case GL_UNSIGNED_INT_8_8_8_8_REV:
#endif
                *type = GL_UNSIGNED_BYTE;
                // fall through
            case GL_UNSIGNED_BYTE:
                if (dest_format == GL_RGB && globals4es.avoid24bits) {
                    dest_format = GL_RGBA;
                    convert = 1;
                }
                break;
#ifdef __BIG_ENDIAN__
            case GL_UNSIGNED_INT_8_8_8_8_REV:
                if (!hardext.rgba8888rev) {
                    dest_type = GL_UNSIGNED_BYTE;
                    convert = 1;
                }
                break;
#else
            case GL_UNSIGNED_INT_8_8_8_8:
                if (!hardext.rgba8888) {
                    dest_type = GL_UNSIGNED_BYTE;
                    convert = 1;
                }
                break;
#endif
            case GL_UNSIGNED_INT_24_8:
                if (hardext.depthtex && hardext.depthstencil) {
                    dest_type = GL_UNSIGNED_INT_24_8;
                } else {
                    *type = GL_UNSIGNED_BYTE; // will probably do nothing good!
                    convert = 1;
                }
                break;
            case GL_FLOAT_32_UNSIGNED_INT_24_8_REV:
                if (hardext.floattex && hardext.depthstencil) {
                    dest_type = GL_FLOAT_32_UNSIGNED_INT_24_8_REV;
                } else {
                    *type = GL_UNSIGNED_BYTE; // will probably do nothing good!
                    convert = 1;
                }
                break;
            case GL_FLOAT:
                if (hardext.floattex)
                    dest_type = GL_FLOAT;
                else
                    convert = 1;
                break;
            case GL_HALF_FLOAT:
            case GL_HALF_FLOAT_OES:
                if (hardext.halffloattex)
                    dest_type = GL_HALF_FLOAT_OES;
                else
                    convert = 1;
                break;
            default:
                convert = 1;
                break;
            }
    }
    if (data) {
        if (convert) {
            GLvoid* pixels = (GLvoid*)data;
            bound->inter_format = dest_format;
            bound->format = dest_format;
            bound->inter_type = dest_type;
            bound->type = dest_type;
            if (!pixel_convert(data, &pixels, width, height, *format, *type, dest_format, dest_type, 0,
                               glstate->texture.unpack_align)) {
                DBGLOGD("LIBGL: swizzle error: (%s, %s -> %s, %s)\n", PrintEnum(*format), PrintEnum(*type),
                        PrintEnum(dest_format), PrintEnum(dest_type));
                return NULL;
            }
            *type = dest_type;
            *format = dest_format;
            if (dest_format != internalformat) {
                GLvoid* pix2 = (GLvoid*)pixels;
                internal2format_type(&internalformat, &dest_format, &dest_type);
                bound->format = dest_format;
                bound->type = dest_type;
                if (!pixel_convert(pixels, &pix2, width, height, *format, *type, dest_format, dest_type, 0,
                                   glstate->texture.unpack_align)) {
                    DBGLOGD("LIBGL: swizzle error: (%s, %s -> %s, %s)\n", PrintEnum(dest_format), PrintEnum(dest_type),
                            PrintEnum(internalformat), PrintEnum(dest_type));
                    return NULL;
                }
                if (pix2 != pixels) {
                    if (pixels != data) free(pixels);
                    pixels = pix2;
                }
                *type = dest_type;
                *format = dest_format;
            }
            GLvoid* pix2 = pixels;
            if (raster_need_transform())
                if (!pixel_transform(data, &pixels, width, height, *format, *type, glstate->raster.raster_scale,
                                     glstate->raster.raster_bias)) {
                    DBGLOGD("LIBGL: swizzle/conver