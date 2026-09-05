# Unequal-dimension Gaussian interchange fixture

## GNNSection
ActInfContinuous

## ModelName
Rectangular Gaussian interchange

## StateSpaceBlock
x[3,1,type=float]
y[2,1,type=float]
u[1,1,type=float]
F[3,3,type=float]
G[3,1,type=float]
H[2,3,type=float]
Q[3,3,type=float]
R[2,2,type=float]
prior_mean[3,type=float]
prior_cov[3,3,type=float]

## InitialParameterization
F={(2.0,0.0,0.0),(0.0,1.0,0.0),(0.0,0.0,0.5)}
G={(1.0,),(2.0,),(0.0,)}
H={(1.0,0.0,0.0),(0.0,1.0,0.0)}
Q={(0.1,0.0,0.0),(0.0,0.1,0.0),(0.0,0.0,0.1)}
R={(0.5,0.0),(0.0,2.0)}
prior_mean={(0.0,0.0,0.0)}
prior_cov={(1.0,0.0,0.0),(0.0,4.0,0.0),(0.0,0.0,9.0)}

## Time
Dynamic
Discrete
