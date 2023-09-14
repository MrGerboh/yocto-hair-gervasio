#include "yocto_hair.h"
#include <yocto/yocto_shading.h>

#include <atomic>
#include <deque>
#include <future>
#include <memory>
#include <mutex>
#include <stdexcept>

using namespace std::string_literals;

// -----------------------------------------------------------------------------
// MATH FUNCTIONS
// -----------------------------------------------------------------------------
namespace yocto::hair {

// import math symbols for use
using yocto::abs;
using yocto::acos;
using yocto::atan2;
using yocto::clamp;
using yocto::cos;
using yocto::exp;
using yocto::flt_eps;
using yocto::flt_max;
using yocto::fmod;
using yocto::fresnel_conductor;
using yocto::fresnel_dielectric;
using yocto::identity3x3f;
using yocto::invalidb3f;
using yocto::log;
using yocto::luminance;
using yocto::make_rng;
using yocto::max;
using yocto::min;
using yocto::pif;
using yocto::pow;
using yocto::rand1f;
using yocto::rand2f;
using yocto::sample_sphere;
using yocto::sample_sphere_pdf;
using yocto::sample_uniform;
using yocto::sample_uniform_pdf;
using yocto::sin;
using yocto::sqrt;
using yocto::zero2f;
using yocto::zero2i;
using yocto::zero3f;
using yocto::zero3i;
using yocto::zero4f;
using yocto::zero4i;

}  // namespace yocto::hair

namespace yocto::hair {

inline float Sqr(float v) { return v * v; }

inline vec3f Sqr(vec3f v) { return v * v; }

inline float SafeSqrt(float x) { return sqrt(max(0.0f, x)); }

inline float SafeAsin(float x) { return asin(clamp(x, -1.0f, 1.0f)); }

template <int n>
static float Pow(float v) {
  auto n2 = Pow<n / 2>(v);
  return n2 * n2 * Pow<n & 1>(v);
}

template <>
inline float Pow<1>(float v) {
  return v;
}

template <>
inline float Pow<0>(float v) {
  return 1;
}

inline const float sqrt_pi_over_8f = 0.626657069f;

inline float I0(float x) {
  float   val   = 0;
  float   x2i   = 1;
  int64_t ifact = 1;
  int     i4    = 1;
  // I0(x) \approx Sum_i x^(2i) / (4^i (i!)^2)
  for (int i = 0; i < 10; i++) {
    if (i > 1) ifact *= i;
    auto sqrIfact = ifact * ifact;
    val += x2i / (i4 * sqrIfact);
    x2i *= x * x;
    i4 *= 4;
  }
  return val;
}

inline float LogI0(float x) {
  if (x > 12)
    return x + 0.5f * (-log(2 * pif) + log(1 / x) + 1 / (8 * x));
  else
    return log(I0(x));
}

static float Mp(float cosThetaI, float cosThetaO, float sinThetaI,
    float sinThetaO, float v) {
  auto a = cosThetaI * cosThetaO / v;
  auto b = sinThetaI * sinThetaO / v;
  return (v <= 0.1f) ? (exp(LogI0(a) - b - 1 / v + 0.6931f + log(1 / (2 * v))))
                     : (exp(-b) * I0(a)) / (sinh(1 / v) * 2 * v);
}

static std::array<vec3f, pMax + 1> Ap(
    float cosThetaO, float eta, float h, const vec3f& T) {
  auto ap = std::array<vec3f, pMax + 1>{};
  // Compute $p=0$ attenuation at initial cylinder intersection
  float cosGammaO = SafeSqrt(1 - h * h);
  float cosTheta  = cosThetaO * cosGammaO;

  // We force two vectors s.t. their dot product is equal to cosTheta
  auto f = fresnel_dielectric(eta, {0, 0, 1}, {0, 0, cosTheta});
  ap[0]  = vec3f{f, f, f};

  // Compute $p=1$ attenuation term
  ap[1] = Sqr(1 - f) * T;

  // Compute attenuation terms up to $p=_pMax_$
  for (auto p = 2; p < pMax; p++) ap[p] = ap[p - 1] * T * f;

  // Compute attenuation term accounting for remaining orders of scattering
  ap[pMax] = ap[pMax - 1] * f * T / (vec3f{1.f, 1.f, 1.f} - T * f);
  return ap;
}

inline float Phi(int p, float gammaO, float gammaT) {
  return 2 * p * gammaT - 2 * gammaO + p * pif;
}

inline float Logistic(float x, float s) {
  x = abs(x);
  return exp(-x / s) / (s * Sqr(1 + exp(-x / s)));
}

inline float LogisticCDF(float x, float s) { return 1 / (1 + exp(-x / s)); }

inline float TrimmedLogistic(float x, float s, float a, float b) {
  return Logistic(x, s) / (LogisticCDF(b, s) - LogisticCDF(a, s));
}

inline float Np(float phi, int p, float s, float gammaO, float gammaT) {
  auto dphi = phi - Phi(p, gammaO, gammaT);
  // Remap _dphi_ to $[-\pi,\pi]$
  while (dphi > pif) dphi -= 2 * pif;
  while (dphi < -pif) dphi += 2 * pif;
  return TrimmedLogistic(dphi, s, -pif, pif);
}

static float SampleTrimmedLogistic(float u, float s, float a, float b) {
  auto k = LogisticCDF(b, s) - LogisticCDF(a, s);
  auto x = -s * log(1 / (u * k + LogisticCDF(a, s)) - 1);
  return clamp(x, a, b);
}

static vec3f sigmaAFromConcentration(float ce, float cp) {
  auto eumelaninSigmaA   = vec3f{0.419f, 0.697f, 1.37f};
  auto pheomelaninSigmaA = vec3f{0.187f, 0.4f, 1.05f};
  return ce * eumelaninSigmaA + cp * pheomelaninSigmaA;
}

static vec3f sigmaAFromReflectance(const vec3f& c, float beta_n) {
  return Sqr(log(c) / (5.969f - 0.215f * beta_n + 2.532f * Sqr(beta_n) -
                          10.73f * Pow<3>(beta_n) + 5.574f * Pow<4>(beta_n) +
                          0.245f * Pow<5>(beta_n)));
}

hair_brdf eval_hair_brdf(const hair_material& material, float v,
    const vec3f& normal, const vec3f& tangent) {
  auto brdf = hair_brdf{};

  if (material.sigmaA.x > 0 || material.sigmaA.y > 0 || material.sigmaA.z > 0) {
    brdf.sigmaA = material.sigmaA;
  } else if (material.color.x > 0 || material.color.y > 0 ||
             material.color.z > 0) {
    brdf.sigmaA = sigmaAFromReflectance(material.color, material.betaN);
  } else if (material.eumelanin || material.pheomelanin) {
    brdf.sigmaA = sigmaAFromConcentration(
        material.eumelanin, material.pheomelanin);
  }

  auto beta_m = material.betaM;
  auto beta_n = material.betaN;
  brdf.alpha  = material.alpha;
  brdf.eta    = material.eta;

  brdf.h = -1 + 2 * v;

  brdf.gammaO = SafeAsin(brdf.h);

  // Compute longitudinal variance from $\beta_m$
  brdf.v[0] = Sqr(
      0.726f * beta_m + 0.812f * Sqr(beta_m) + 3.7f * Pow<20>(beta_m));
  brdf.v[1] = 0.25f * brdf.v[0];
  brdf.v[2] = 4 * brdf.v[0];
  for (auto p = 3; p <= pMax; p++) brdf.v[p] = brdf.v[2];

  // Compute azimuthal logistic scale factor from $\beta_n$
  brdf.s = sqrt_pi_over_8f *
           (0.265f * beta_n + 1.194f * Sqr(beta_n) + 5.372f * Pow<22>(beta_n));

  // Compute $\alpha$ terms for hair scales
  brdf.sin_2k_alpha[0] = sin(pif / 180 * brdf.alpha);
  brdf.cos_2k_alpha[0] = SafeSqrt(1 - Sqr(brdf.sin_2k_alpha[0]));
  for (auto i = 1; i < 3; i++) {
    brdf.sin_2k_alpha[i] = 2 * brdf.cos_2k_alpha[i - 1] *
                           brdf.sin_2k_alpha[i - 1];
    brdf.cos_2k_alpha[i] = Sqr(brdf.cos_2k_alpha[i - 1]) -
                           Sqr(brdf.sin_2k_alpha[i - 1]);
  }

  brdf.world_to_brdf = inverse(frame_fromzx(zero3f, normal, tangent));

  return brdf;
}

vec3f eval_hair_scattering(
    const hair_brdf& brdf, const vec3f& outgoing_, const vec3f& incoming_) {
  auto sigma_a       = brdf.sigmaA;
  auto eta           = brdf.eta;
  auto h             = brdf.h;
  auto gammaO        = brdf.gammaO;
  auto v             = brdf.v;
  auto s             = brdf.s;
  auto sin_2k_alpha  = brdf.sin_2k_alpha;
  auto cos_2k_alpha  = brdf.cos_2k_alpha;
  auto world_to_brdf = brdf.world_to_brdf;

  auto outgoing = transform_direction(world_to_brdf, outgoing_);
  auto incoming = transform_direction(world_to_brdf, incoming_);

  // Compute hair coordinate system terms related to _wo_
  auto sinThetaO = outgoing.x;
  auto cosThetaO = SafeSqrt(1 - Sqr(sinThetaO));
  auto phi_o     = atan2(outgoing.z, outgoing.y);

  // Compute hair coordinate system terms related to _wi_
  auto sinThetaI = incoming.x;
  auto cosThetaI = SafeSqrt(1 - Sqr(sinThetaI));
  auto phi_i     = atan2(incoming.z, incoming.y);

  // Compute $\cos \thetat$ for refracted ray
  auto sinThetaT = sinThetaO / eta;
  auto cosThetaT = SafeSqrt(1 - Sqr(sinThetaT));

  // Compute $\gammat$ for refracted ray
  auto etap      = sqrt(eta * eta - Sqr(sinThetaO)) / cosThetaO;
  auto sinGammaT = h / etap;
  auto cosGammaT = SafeSqrt(1 - Sqr(sinGammaT));
  auto gamma_t   = SafeAsin(sinGammaT);

  // Compute the transmittance _T_ of a single path through the cylinder
  auto T = exp(-sigma_a * (2 * cosGammaT / cosThetaT));

  // Evaluate hair BSDF
  auto phi  = phi_i - phi_o;
  auto ap   = Ap(cosThetaO, eta, h, T);
  auto fsum = zero3f;
  for (auto p = 0; p < pMax; p++) {
    // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for scales
    auto sinThetaOp = 0.0f;
    auto cosThetaOp = 0.0f;
    if (p == 0) {
      sinThetaOp = sinThetaO * cos_2k_alpha[1] - cosThetaO * sin_2k_alpha[1];
      cosThetaOp = cosThetaO * cos_2k_alpha[1] + sinThetaO * sin_2k_alpha[1];
    }

    // Handle remainder of $p$ values for hair scale tilt
    else if (p == 1) {
      sinThetaOp = sinThetaO * cos_2k_alpha[0] + cosThetaO * sin_2k_alpha[0];
      cosThetaOp = cosThetaO * cos_2k_alpha[0] - sinThetaO * sin_2k_alpha[0];
    } else if (p == 2) {
      sinThetaOp = sinThetaO * cos_2k_alpha[2] + cosThetaO * sin_2k_alpha[2];
      cosThetaOp = cosThetaO * cos_2k_alpha[2] - sinThetaO * sin_2k_alpha[2];
    } else {
      sinThetaOp = sinThetaO;
      cosThetaOp = cosThetaO;
    }

    // Handle out-of-range $\cos \thetao$ from scale adjustment
    cosThetaOp = abs(cosThetaOp);
    fsum += Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, v[p]) * ap[p] *
            Np(phi, p, s, gammaO, gamma_t);
  }

  // Compute contribution of remaining terms after _pMax_
  fsum += Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, v[pMax]) * ap[pMax] /
          (2 * pif);

  return fsum;
}

// https://fgiesen.wordpress.com/2009/12/13/decoding-morton-codes/
static uint32_t Compact1By1(uint32_t x) {
  // x = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
  x &= 0x55555555;
  // x = --fe --dc --ba --98 --76 --54 --32 --10
  x = (x ^ (x >> 1)) & 0x33333333;
  // x = ---- fedc ---- ba98 ---- 7654 ---- 3210
  x = (x ^ (x >> 2)) & 0x0f0f0f0f;
  // x = ---- ---- fedc ba98 ---- ---- 7654 3210
  x = (x ^ (x >> 4)) & 0x00ff00ff;
  // x = ---- ---- ---- ---- fedc ba98 7654 3210
  x = (x ^ (x >> 8)) & 0x0000ffff;
  return x;
}

static vec2f DemuxFloat(float f) {
  uint64_t v       = f * (1ull << 32);
  uint32_t bits[2] = {Compact1By1(v), Compact1By1(v >> 1)};
  return {bits[0] / float(1 << 16), bits[1] / float(1 << 16)};
}

static std::array<float, pMax + 1> ComputeApPdf(
    const hair_brdf& brdf, float cosThetaO) {
  auto sigma_a = brdf.sigmaA;
  auto eta     = brdf.eta;
  auto h       = brdf.h;

  // Compute array of $A_p$ values for _cosThetaO_
  auto sinThetaO = SafeSqrt(1 - cosThetaO * cosThetaO);

  // Compute $\cos \thetat$ for refracted ray
  auto sinThetaT = sinThetaO / eta;
  auto cosThetaT = SafeSqrt(1 - Sqr(sinThetaT));

  // Compute $\gammat$ for refracted ray
  auto etap      = sqrt(eta * eta - Sqr(sinThetaO)) / cosThetaO;
  auto sinGammaT = h / etap;
  auto cosGammaT = SafeSqrt(1 - Sqr(sinGammaT));

  // Compute the transmittance _T_ of a single path through the cylinder
  auto T  = exp(-sigma_a * (2 * cosGammaT / cosThetaT));
  auto ap = Ap(cosThetaO, eta, h, T);

  // Compute $A_p$ PDF from individual $A_p$ terms
  auto ap_pdf = std::array<float, pMax + 1>{};
  auto sum_y  = 0.0f;
  for (auto i = 0; i <= pMax; i++) {
    sum_y += luminance(ap[i]);
  }
  for (auto i = 0; i <= pMax; i++) {
    ap_pdf[i] = luminance(ap[i]) / sum_y;
  }
  return ap_pdf;
}

vec3f sample_hair_scattering(
    const hair_brdf& brdf, const vec3f& outgoing_, const vec2f& rn) {
  auto eta           = brdf.eta;
  auto h             = brdf.h;
  auto gammaO        = brdf.gammaO;
  auto v             = brdf.v;
  auto s             = brdf.s;
  auto sin_2k_alpha  = brdf.sin_2k_alpha;
  auto cos_2k_alpha  = brdf.cos_2k_alpha;
  auto world_to_brdf = brdf.world_to_brdf;

  auto outgoing = transform_direction(world_to_brdf, outgoing_);

  // Compute hair coordinate system terms related to _wo_
  auto sinThetaO = outgoing.x;
  auto cosThetaO = SafeSqrt(1 - Sqr(sinThetaO));
  auto phi_o     = atan2(outgoing.z, outgoing.y);

  // Derive four random samples from _u2_
  auto u = std::array<vec2f, 2>{DemuxFloat(rn.x), DemuxFloat(rn.y)};

  // Determine which term $p$ to sample for hair scattering
  auto ap_pdf = ComputeApPdf(brdf, cosThetaO);
  auto p      = 0;
  for (p = 0; p < pMax; p++) {
    if (u[0][0] < ap_pdf[p]) break;
    u[0][0] -= ap_pdf[p];
  }

  // Rotate $\sin \thetao$ and $\cos \thetao$ to account for hair scale tilt
  auto sinThetaOp = 0.0f;
  auto cosThetaOp = 0.0f;
  if (p == 0) {
    sinThetaOp = sinThetaO * cos_2k_alpha[1] - cosThetaO * sin_2k_alpha[1];
    cosThetaOp = cosThetaO * cos_2k_alpha[1] + sinThetaO * sin_2k_alpha[1];
  } else if (p == 1) {
    sinThetaOp = sinThetaO * cos_2k_alpha[0] + cosThetaO * sin_2k_alpha[0];
    cosThetaOp = cosThetaO * cos_2k_alpha[0] - sinThetaO * sin_2k_alpha[0];
  } else if (p == 2) {
    sinThetaOp = sinThetaO * cos_2k_alpha[2] + cosThetaO * sin_2k_alpha[2];
    cosThetaOp = cosThetaO * cos_2k_alpha[2] - sinThetaO * sin_2k_alpha[2];
  } else {
    sinThetaOp = sinThetaO;
    cosThetaOp = cosThetaO;
  }

  // Sample $M_p$ to compute $\thetai$
  u[1][0]        = max(u[1][0], 1e-5f);
  auto cosTheta  = 1 + v[p] * log(u[1][0] + (1 - u[1][0]) * exp(-2 / v[p]));
  auto sinTheta  = SafeSqrt(1 - Sqr(cosTheta));
  auto cosPhi    = cos(2 * pif * u[1][1]);
  auto sinThetaI = -cosTheta * sinThetaOp + sinTheta * cosPhi * cosThetaOp;
  auto cosThetaI = SafeSqrt(1 - Sqr(sinThetaI));

  // Sample $N_p$ to compute $\Delta\phi$

  // Compute $\gammat$ for refracted ray
  auto etap        = sqrt(eta * eta - Sqr(sinThetaO)) / cosThetaO;
  auto sin_gamma_t = h / etap;
  auto gamma_t     = SafeAsin(sin_gamma_t);
  auto dphi        = 0.0f;
  if (p < pMax)
    dphi = Phi(p, gammaO, gamma_t) +
           SampleTrimmedLogistic(u[0][1], s, -pif, pif);
  else
    dphi = 2 * pif * u[0][1];

  // Compute _wi_ from sampled hair scattering angles
  auto phi_i = phi_o + dphi;

  auto incoming = vec3f{
      sinThetaI, cosThetaI * cos(phi_i), cosThetaI * sin(phi_i)};
  return transform_direction(inverse(world_to_brdf), incoming);
}

float sample_hair_scattering_pdf(
    const hair_brdf& brdf, const vec3f& outgoing_, const vec3f& incoming_) {
  auto eta           = brdf.eta;
  auto h             = brdf.h;
  auto gammaO        = brdf.gammaO;
  auto v             = brdf.v;
  auto s             = brdf.s;
  auto sin_2k_alpha  = brdf.sin_2k_alpha;
  auto cos_2k_alpha  = brdf.cos_2k_alpha;
  auto world_to_brdf = brdf.world_to_brdf;

  auto outgoing = transform_direction(world_to_brdf, outgoing_);
  auto incoming = transform_direction(world_to_brdf, incoming_);

  // Compute hair coordinate system terms related to _wo_
  auto sinThetaO = outgoing.x;
  auto cosThetaO = SafeSqrt(1 - Sqr(sinThetaO));
  auto phi_o     = atan2(outgoing.z, outgoing.y);

  // Compute hair coordinate system terms related to _wi_
  auto sinThetaI = incoming.x;
  auto cosThetaI = SafeSqrt(1 - Sqr(sinThetaI));
  auto phi_i     = atan2(incoming.z, incoming.y);

  // Compute $\gammat$ for refracted ray
  auto etap        = sqrt(eta * eta - Sqr(sinThetaO)) / cosThetaO;
  auto sin_gamma_t = h / etap;
  auto gamma_t     = SafeAsin(sin_gamma_t);

  // Compute PDF for $A_p$ terms
  auto ap_pdf = ComputeApPdf(brdf, cosThetaO);

  // Compute PDF sum for hair scattering events
  auto phi = phi_i - phi_o;
  auto pdf = 0.0f;
  for (auto p = 0; p < pMax; p++) {
    // Compute $\sin \thetao$ and $\cos \thetao$ terms accounting for scales
    auto sinThetaOp = 0.0f;
    auto cosThetaOp = 0.0f;
    if (p == 0) {
      sinThetaOp = sinThetaO * cos_2k_alpha[1] - cosThetaO * sin_2k_alpha[1];
      cosThetaOp = cosThetaO * cos_2k_alpha[1] + sinThetaO * sin_2k_alpha[1];
    }

    // Handle remainder of $p$ values for hair scale tilt
    else if (p == 1) {
      sinThetaOp = sinThetaO * cos_2k_alpha[0] + cosThetaO * sin_2k_alpha[0];
      cosThetaOp = cosThetaO * cos_2k_alpha[0] - sinThetaO * sin_2k_alpha[0];
    } else if (p == 2) {
      sinThetaOp = sinThetaO * cos_2k_alpha[2] + cosThetaO * sin_2k_alpha[2];
      cosThetaOp = cosThetaO * cos_2k_alpha[2] - sinThetaO * sin_2k_alpha[2];
    } else {
      sinThetaOp = sinThetaO;
      cosThetaOp = cosThetaO;
    }

    // Handle out-of-range $\cos \thetao$ from scale adjustment
    cosThetaOp = abs(cosThetaOp);
    pdf += Mp(cosThetaI, cosThetaOp, sinThetaI, sinThetaOp, v[p]) * ap_pdf[p] *
           Np(phi, p, s, gammaO, gamma_t);
  }
  pdf += Mp(cosThetaI, cosThetaO, sinThetaI, sinThetaO, v[pMax]) *
         ap_pdf[pMax] * (1 / (2 * pif));
  return pdf;
}

void white_furnace_test() {
  auto rng = make_rng(199382389514);
  auto wo  = sample_sphere(rand2f(rng));
  for (auto betaM = 0.1f; betaM < 1.0f; betaM += 0.2f) {
    for (auto betaN = 0.1f; betaN < 1.0f; betaN += 0.2f) {
      // Estimate reflected uniform incident radiance from hair
      auto sum   = zero3f;
      auto count = 300000;
      for (auto i = 0; i < count; i++) {
#ifdef YOCTO_EMBREE
        auto h = -1 + 2 * rand1f(rng);
#else
        auto h = rand1f(rng);
#endif
        // the original pbrt test fails when h == 0
        if (h == 0) h += flt_eps;

        auto mat   = hair_material{};
        mat.betaM  = betaM;
        mat.betaN  = betaN;
        mat.alpha  = 0;
        auto brdf  = eval_hair_brdf(mat, h, {0, 0, 1}, {1, 0, 0});
        auto wi    = sample_sphere(rand2f(rng));
        sum += eval_hair_scattering(brdf, wo, wi);
      }
      auto avg = luminance(sum) / (count * sample_sphere_pdf(wo));
      if (!(avg >= 0.95f && avg <= 1.05f))
        throw std::runtime_error("TEST FAILED!");
    }
  }
  printf("OK!\n");
  fflush(stdout);
}

void white_furnace_sampled_test() {
  auto rng = make_rng(199382389514);
  auto wo  = sample_sphere(rand2f(rng));
  for (auto betaM = 0.1f; betaM < 1.0f; betaM += 0.2f) {
    for (auto betaN = 0.1f; betaN < 1.0f; betaN += 0.2f) {
      auto sum   = zero3f;
      auto count = 300000;
      for (auto i = 0; i < count; i++) {
#ifdef YOCTO_EMBREE
        auto h = -1 + 2 * rand1f(rng);
#else
        auto h = rand1f(rng);
#endif
        auto mat   = hair_material{};
        mat.betaM  = betaM;
        mat.betaN  = betaN;
        mat.alpha  = 0;
        auto brdf  = eval_hair_brdf(mat, h, {0, 0, 1}, {1, 0, 0});
        auto wi    = sample_hair_scattering(brdf, wo, rand2f(rng));
        auto f     = eval_hair_scattering(brdf, wo, wi);
        auto pdf   = sample_hair_scattering_pdf(brdf, wo, wi);
        if (pdf > 0) sum += f / pdf;
      }
      auto avg = luminance(sum) / (count);
      if (!(avg >= 0.99f && avg <= 1.01f))
        throw std::runtime_error("TEST FAILED!");
    }
  }
  printf("OK!\n");
  fflush(stdout);
}

void sampling_weights_test() {
  auto rng = make_rng(199382389514);
  for (auto betaM = 0.1f; betaM < 1.0f; betaM += 0.2f) {
    for (auto betaN = 0.4f; betaN < 1.0f; betaN += 0.2f) {
      auto count = 10000;
      for (auto i = 0; i < count; i++) {
        // Check _HairBSDF::Sample\_f()_ sample weight
#ifdef YOCTO_EMBREE
        auto h = -1 + 2 * rand1f(rng);
#else
        auto h = rand1f(rng);
#endif
        auto mat   = hair_material{};
        mat.betaM  = betaM;
        mat.betaN  = betaN;
        mat.alpha  = 0;
        auto brdf  = eval_hair_brdf(mat, h, {0, 0, 1}, {1, 0, 0});
        auto wo    = sample_sphere(rand2f(rng));
        auto wi    = sample_hair_scattering(brdf, wo, rand2f(rng));
        auto f     = eval_hair_scattering(brdf, wo, wi);
        auto pdf   = sample_hair_scattering_pdf(brdf, wo, wi);
        if (pdf > 0) {
          // Verify that hair BSDF sample weight is close to 1 for _wi_
          if (!(luminance(f) / pdf >= 0.999f && luminance(f) / pdf <= 1.001f))
            throw std::runtime_error("TEST FAILED!");
        }
      }
    }
  }
  printf("OK!\n");
  fflush(stdout);
}

void sampling_consistency_test() {
  auto rng = make_rng(199382389514);
  for (auto betaM = 0.2f; betaM < 1.0f; betaM += 0.2f)
    for (auto betaN = 0.4f; betaN < 1.0f; betaN += 0.2f) {
      // Declare variables for hair sampling test
      const auto count   = 64 * 1024;
      auto       sigma_a = vec3f{0.25f, 0.25f, 0.25f};
      auto       wo      = sample_sphere(rand2f(rng));
      auto       li      = [](const vec3f& w) -> vec3f {
        return vec3f{w.z * w.z, w.z * w.z, w.z * w.z};
      };
      auto       f_importance = zero3f;
      auto       f_uniform    = zero3f;
      for (auto i = 0; i < count; i++) {
        // Compute estimates of scattered radiance for hair sampling test
#ifdef YOCTO_EMBREE
        auto h = -1 + 2 * rand1f(rng);
#else
        auto h = rand1f(rng);
#endif
        auto mat   = hair_material{};
        mat.betaM  = betaM;
        mat.betaN  = betaN;
        mat.alpha  = 0;
        auto brdf  = eval_hair_brdf(mat, h, {0, 0, 1}, {1, 0, 0});
        auto u     = rand2f(rng);
        auto wi    = sample_hair_scattering(brdf, wo, u);
        auto f     = eval_hair_scattering(brdf, wo, wi);
        auto pdf   = sample_hair_scattering_pdf(brdf, wo, wi);
        if (pdf > 0) f_importance += f * li(wi) / (count * pdf);
        wi = sample_sphere(u);
        f_uniform += eval_hair_scattering(brdf, wo, wi) * li(wi) /
                     (count * sample_sphere_pdf(wo));
      }
      // Verify consistency of estimated hair reflected radiance values
      auto err = abs(luminance(f_importance) - luminance(f_uniform)) /
                 luminance(f_uniform);
      if (err >= 0.05f) throw std::runtime_error("TEST FAILED!");
    }
  printf("OK!\n");
  fflush(stdout);
}

} // namespace yocto::hair