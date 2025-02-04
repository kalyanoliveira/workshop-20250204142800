def sse(po, gto):
    return sum(
        [
            (p - gt)**2 for p, gt in zip (po, gto)
        ]
    )

def total_sse(pos, gtos):
    return sum(
        [
            sse(po, gto) for po, gto in zip(pos, gtos)
        ]
    )

def mse(po, gto):
    return sum(
        [
            (p - gt)**2 for p, gt in zip(po, gto)
        ]
    ) / len(gto)

def avg_mse(pos, gtos):
    return sum(
        [
            mse(po, gto) for po, gto in zip (pos, gtos)
        ]
    ) / len(gtos)
