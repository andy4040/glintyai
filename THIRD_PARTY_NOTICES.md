# Third-party notices

GLINT uses a pinned [gsplat](https://github.com/youngju-na/gsplat) commit based
on upstream commit `77ab983f` as an external Python dependency. gsplat is
distributed under the Apache License 2.0. Its source is not included in this
repository.

This fork vendors `third_party/diff-surfel-tracing` from
https://github.com/xbillowy/diff-surfel-tracing at commit
`ef6f24be9174eee037469f7e221cd72f6c0222cb`, with modifications for per-ray
source-surface filtering in forward and backward tracing. Its MIT license is
retained in that directory.

The nested NVIDIA OptiX headers originate from https://github.com/NVIDIA/optix-dev
at commit `7b5c4e8608b8b4b601729f6240fc3fd53cb36d23`. Their original license files
are retained under `third_party/diff-surfel-tracing/third_party/optix` and remain
authoritative for those files. Vendoring ensures this fork's native changes are
available on a fresh clone without an unpublished submodule commit.

Refer to each dependency's license files for the complete and authoritative
terms. This notice is informational and does not replace those terms.
