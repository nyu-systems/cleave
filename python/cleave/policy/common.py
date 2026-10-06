from typing import List
import numpy as np


def get_all_factors(n: int) -> List[int]:
    n0 = int(np.ceil(np.sqrt(n)))
    val = np.where(n % np.arange(1, n0) == 0)[0] + 1
    mid = np.array([], dtype=int) if n0 * n0 != n else [n0]
    return [int(x) for x in np.concatenate([val, mid, n // val[::-1]])]


def get_all_round_factors(n: int) -> List[int]:
    factors, i = [], 0
    while 2**i <= n and n % (2**i) == 0:
        factors.append(2**i)
        i += 1
    if n not in factors:
        factors.append(n)

    # fallback to default
    if factors == []:
        factors = get_all_factors(n)
    return factors


def coalesced_factor(subtensor: List[int], tensor: List[int]) -> int:
    if subtensor[-1] != tensor[-1] or len(subtensor) == 1:
        return subtensor[-1]
    else:
        return subtensor[-1] * coalesced_factor(subtensor[:-1], tensor[:-1])


def coalesced_tensor_shape(
    subtensor: List[int], tensor: List[int], transaction_size: int
) -> int:
    # Calculate the total number of elements in the subtensor
    bytes = int(np.prod(subtensor))

    if bytes == 0:
        return 0

    # Calculate the coalesced factor for the subtensor
    factor = int(coalesced_factor(subtensor, tensor))

    # Compute the shape of the coalesced tensor
    return transaction_size * bytes / min(transaction_size, factor)


def factorize(n: int) -> List[int]:
    i = 2  # Start with the smallest prime number
    result = []

    # Iterate through numbers to find factors
    while n > 1:
        if n % i == 0:  # If i is a factor of n
            n //= i  # Divide n by i and keep the integer part
            result.append(i)
        else:
            i += 1  # Try the next number
    return result
