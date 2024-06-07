- [ ] Scheme:
    1. At timestep 0, learn $G$
    2. At trimestep t != 0:
        1. $f(G, t, \theta) = \Delta$
        2. $y = G + \Delta$
        3. $\mathcal{L}(y, t)$
    3. Optimize:
       1. means3D