"""Binary expansion of a real number, to show why 0.1 is not exact in floating point."""


def convertir_binario(n, lim = 24):
    bits = ""
    ent = n // 1
    n -= ent
    bits = bits + str(ent) + "."
    i = 0
    while n > 0 and i < lim:
        n *= 2
        if n//1 == 1:
            bits = bits + "1"

        else:
            bits = bits + "0"
        i += 1

    return bits
