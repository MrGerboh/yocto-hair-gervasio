#ifndef _YOCTO_HAIR_H_
#define _YOCTO_HAIR_H_

// -----------------------------------------------------------------------------
// INCLUDES
// -----------------------------------------------------------------------------

#include <yocto/yocto_image.h>
#include <yocto/yocto_math.h>
#include <yocto/yocto_sampling.h>
#include <yocto/yocto_geometry.h>
#include <yocto/yocto_scene.h>

#include <atomic>
#include <future>
#include <memory>

namespace yocto {
struct material;
}

// -----------------------------------------------------------------------------
// ALIASES
// -----------------------------------------------------------------------------
namespace yocto::hair {

// Namespace aliases
namespace hair = yocto::hair;

// Math defitions
using yocto::bbox3f;
using yocto::byte;
using yocto::clamp;
using yocto::frame3f;
using yocto::identity3x4f;
using yocto::max;
using yocto::ray3f;
using yocto::rng_state;
using yocto::vec2f;
using yocto::vec2i;
using yocto::vec3f;
using yocto::vec3i;
using yocto::vec4f;
using yocto::vec4i;
using yocto::zero2f;
using yocto::zero3f;

}  // namespace yocto::hair

namespace yocto::hair {

inline const int pMax = 3;

struct hair_material{
  vec3f         sigmaA      = zero3f;
  float         betaM       = 0.3;
  float         betaN       = 0.3;
  float         alpha       = 2;
  float         eta         = 1.55;
  vec3f         color       = zero3f;
  float         eumelanin   = 0;
  float         pheomelanin = 0;
};

struct hair_brdf {
  vec3f sigmaA  = zero3f;
  float alpha   = 2;
  float eta     = 1.55;
  float h       = 0;

  // computed properties
  std::array<float, pMax + 1> v;
  float                        s = 0;
  vec3f                        sin_2k_alpha;
  vec3f                        cos_2k_alpha;
  float                        gammaO = 0;

  // Allow to convert outgoing and incoming directions to BRDF coordinate
  // system
  frame3f world_to_brdf = identity3x4f;
};

hair_brdf eval_hair_brdf(const hair_material& material, float v,
    const vec3f& normal, const vec3f& tangent);

vec3f eval_hair_scattering(
    const hair_brdf& brdf, const vec3f& outgoing, const vec3f& incoming);

vec3f sample_hair_scattering(
    const hair_brdf& brdf, const vec3f& outgoing, const vec2f& rn);

float sample_hair_scattering_pdf(
    const hair_brdf& brdf, const vec3f& outgoing, const vec3f& incoming);

void white_furnace_test();
void sampling_weights_test();
void white_furnace_sampled_test();
void sampling_consistency_test();

}  // namespace yocto::hair

#endif