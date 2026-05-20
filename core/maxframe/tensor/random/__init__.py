# Copyright 1999-2026 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from maxframe.tensor.random.beta import TensorRandBeta, beta
from maxframe.tensor.random.binomial import TensorBinomial, binomial
from maxframe.tensor.random.bytes import bytes
from maxframe.tensor.random.chisquare import TensorChisquareDist, chisquare
from maxframe.tensor.random.choice import TensorChoice, choice
from maxframe.tensor.random.core import (
    RandomState,
    RandomStateField,
    TensorDistribution,
    _random_state,
)
from maxframe.tensor.random.dirichlet import TensorDirichlet, dirichlet
from maxframe.tensor.random.exponential import TensorExponential, exponential
from maxframe.tensor.random.f import TensorFDist, f
from maxframe.tensor.random.gamma import TensorRandGamma, gamma
from maxframe.tensor.random.geometric import TensorGeometric, geometric
from maxframe.tensor.random.gumbel import TensorGumbel, gumbel
from maxframe.tensor.random.hypergeometric import TensorHypergeometric, hypergeometric
from maxframe.tensor.random.laplace import TensorLaplaceDist, laplace
from maxframe.tensor.random.logistic import TensorLogistic, logistic
from maxframe.tensor.random.lognormal import TensorLognormal, lognormal
from maxframe.tensor.random.logseries import TensorLogseries, logseries
from maxframe.tensor.random.multinomial import TensorMultinomial, multinomial
from maxframe.tensor.random.multivariate_normal import (
    TensorMultivariateNormal,
    multivariate_normal,
)
from maxframe.tensor.random.negative_binomial import (
    TensorNegativeBinomial,
    negative_binomial,
)
from maxframe.tensor.random.noncentral_chisquare import (
    TensorNoncentralChisquare,
    noncentral_chisquare,
)
from maxframe.tensor.random.noncentral_f import TensorNoncentralF, noncentral_f
from maxframe.tensor.random.normal import TensorNormal, normal
from maxframe.tensor.random.pareto import TensorPareto, pareto
from maxframe.tensor.random.permutation import TensorPermutation, permutation
from maxframe.tensor.random.poisson import TensorPoisson, poisson
from maxframe.tensor.random.power import TensorRandomPower, power
from maxframe.tensor.random.rand import TensorRand, rand
from maxframe.tensor.random.randint import TensorRandint, randint
from maxframe.tensor.random.randn import TensorRandn, randn
from maxframe.tensor.random.random_integers import TensorRandomIntegers, random_integers
from maxframe.tensor.random.random_sample import TensorRandomSample, random_sample
from maxframe.tensor.random.rayleigh import TensorRayleigh, rayleigh
from maxframe.tensor.random.shuffle import shuffle
from maxframe.tensor.random.standard_cauchy import TensorStandardCauchy, standard_cauchy
from maxframe.tensor.random.standard_exponential import (
    TensorStandardExponential,
    standard_exponential,
)
from maxframe.tensor.random.standard_gamma import TensorStandardGamma, standard_gamma
from maxframe.tensor.random.standard_normal import TensorStandardNormal, standard_normal
from maxframe.tensor.random.standard_t import TensorStandardT, standard_t
from maxframe.tensor.random.triangular import TensorTriangular, triangular
from maxframe.tensor.random.uniform import TensorUniform, uniform
from maxframe.tensor.random.vonmises import TensorVonmises, vonmises
from maxframe.tensor.random.wald import TensorWald, wald
from maxframe.tensor.random.weibull import TensorWeibull, weibull
from maxframe.tensor.random.zipf import TensorZipf, zipf


def _install():
    setattr(RandomState, "rand", rand)
    setattr(RandomState, "randn", randn)
    setattr(RandomState, "randint", randint)
    setattr(RandomState, "random_integers", random_integers)
    setattr(RandomState, "random_sample", random_sample)
    setattr(RandomState, "ranf", random_sample)
    setattr(RandomState, "random", random_sample)
    setattr(RandomState, "sample", random_sample)
    setattr(RandomState, "choice", choice)
    setattr(RandomState, "bytes", bytes)
    setattr(RandomState, "beta", beta)
    setattr(RandomState, "binomial", binomial)
    setattr(RandomState, "chisquare", chisquare)
    setattr(RandomState, "dirichlet", dirichlet)
    setattr(RandomState, "exponential", exponential)
    setattr(RandomState, "f", f)
    setattr(RandomState, "gamma", gamma)
    setattr(RandomState, "geometric", geometric)
    setattr(RandomState, "gumbel", gumbel)
    setattr(RandomState, "hypergeometric", hypergeometric)
    setattr(RandomState, "laplace", laplace)
    setattr(RandomState, "logistic", logistic)
    setattr(RandomState, "lognormal", lognormal)
    setattr(RandomState, "logseries", logseries)
    setattr(RandomState, "multinomial", multinomial)
    setattr(RandomState, "multivariate_normal", multivariate_normal)
    setattr(RandomState, "negative_binomial", negative_binomial)
    setattr(RandomState, "noncentral_chisquare", noncentral_chisquare)
    setattr(RandomState, "noncentral_f", noncentral_f)
    setattr(RandomState, "normal", normal)
    setattr(RandomState, "pareto", pareto)
    setattr(RandomState, "poisson", poisson)
    setattr(RandomState, "power", power)
    setattr(RandomState, "rayleigh", rayleigh)
    setattr(RandomState, "standard_cauchy", standard_cauchy)
    setattr(RandomState, "standard_exponential", standard_exponential)
    setattr(RandomState, "standard_gamma", standard_gamma)
    setattr(RandomState, "standard_normal", standard_normal)
    setattr(RandomState, "standard_t", standard_t)
    setattr(RandomState, "triangular", triangular)
    setattr(RandomState, "uniform", uniform)
    setattr(RandomState, "vonmises", vonmises)
    setattr(RandomState, "wald", wald)
    setattr(RandomState, "weibull", weibull)
    setattr(RandomState, "zipf", zipf)
    setattr(RandomState, "permutation", permutation)
    setattr(RandomState, "shuffle", shuffle)


_install()
del _install


seed = _random_state.seed

rand = _random_state.rand
randn = _random_state.randn
randint = _random_state.randint
random_integers = _random_state.random_integers
random_sample = _random_state.random_sample
random = _random_state.random
ranf = _random_state.ranf
sample = _random_state.sample
choice = _random_state.choice
bytes = _random_state.bytes

permutation = _random_state.permutation
shuffle = _random_state.shuffle

beta = _random_state.beta
binomial = _random_state.binomial
chisquare = _random_state.chisquare
dirichlet = _random_state.dirichlet
exponential = _random_state.exponential
f = _random_state.f
gamma = _random_state.gamma
geometric = _random_state.geometric
gumbel = _random_state.gumbel
hypergeometric = _random_state.hypergeometric
laplace = _random_state.laplace
logistic = _random_state.logistic
lognormal = _random_state.lognormal
logseries = _random_state.logseries
multinomial = _random_state.multinomial
multivariate_normal = _random_state.multivariate_normal
negative_binomial = _random_state.negative_binomial
noncentral_chisquare = _random_state.noncentral_chisquare
noncentral_f = _random_state.noncentral_f
normal = _random_state.normal
pareto = _random_state.pareto
poisson = _random_state.poisson
power = _random_state.power
rayleigh = _random_state.rayleigh
standard_cauchy = _random_state.standard_cauchy
standard_exponential = _random_state.standard_exponential
standard_gamma = _random_state.standard_gamma
standard_normal = _random_state.standard_normal
standard_t = _random_state.standard_t
triangular = _random_state.triangular
uniform = _random_state.uniform
vonmises = _random_state.vonmises
wald = _random_state.wald
weibull = _random_state.weibull
zipf = _random_state.zipf
