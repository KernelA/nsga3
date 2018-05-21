from problems import ZDT1
from nsga3 import NSGA3
from mutation import PolynomialMutationBound
from crossover import SBXBound


nsga = NSGA3(SBXBound(1, 2), PolynomialMutationBound(0.5, 1))
zdt1 = ZDT1(10)

nsga.minimize(20, zdt1)


