# sgemm

This repository is meant to isolate sgemm from
[browsermt/marian-dev](https://github.com/browsermt/marian-dev).

Archival when integration at the target repository is complete.

Objectives:

- Inputs in marian format - `marian::Tensor`. A bare minimum stub is added
  here.
- Relay into sgemm provider implementing BLAS SGEMM API.
- Benchmark different providers on possible hardware.

From [LAPACK](http://www.netlib.org/lapack/explore-html/db/dc9/group__single__blas__level3_gafe51bacb54592ff5de056acabd83c260.html):

```
SGEMM  performs one of the matrix-matrix operations

    C := alpha*op( A )*op( B ) + beta*C,

 where  op( X ) is one of

    op( X ) = X   or   op( X ) = X**T,
```
