# draw a (deterministic) gaussian noise image
def gaussian_image_drawer(
		s,     # shape tuple
		μ,     # mean
		σ,     # variance
		r      # random seed
		):
	import os, iio
	os.environ["SRAND"] = f"{r}"
	if σ > 0:
		x = iio.read(f"TRANS[pipe=plambda randg]:zero:{s[0]}x{s[1]}")
	else:
		x = iio.read(f"zero:{s[0]}x{s[1]}")
	return μ + σ*x


# Kadkhodaie Simoncelli dynamics
# "Solving Linear Inverse Problems Using the Prior Implicit in a Denoiser"
# Implementation of Algorithm 1 from https://arxiv.org/abs/2007.13640
# The arguments are the same as in the paper, except for "n", a hard-limit
# for the total number of iterations, which is added here for convenience.
def kadkhodaie_simoncelli_algorithm_1(
		s,     # image shape (2 or 3-tuple)
		D,     # denoiser
		σ_0,   # starting sigma
		σ_L,   # last sigma
		h_0,   # starting h
		β,     # neg-temperature (set to 1 for pure hallucination)
		n      # hard limit for the number of iterations
		):
	σ = σ_0
	t = 1
	y = gaussian_image_drawer(s, 0.5, σ ** 2, t-1)
	while σ > σ_L and t < n:
		h = h_0 * t / (1 + h_0 * (t - 1))
		d = D(y) - y
		σ = (d.flatten().T @ d.flatten() / d.size)**0.5;
		γ = ((1 - β*h)**2 - (1 - h)**2)**0.5 * σ
		z = gaussian_image_drawer(s, 0, 1, t)
		y = y + h*d + γ*z
		print(f"t={t} σ={σ} h={h} γ={γ}")
		t = t + 1
	return y

# Implementation of Algorithm 2 from Kadkhodaie-Simoncelli's paper
def kadkhodaie_simoncelli_algorithm_2(
		s,     # image shape (2 or 3-tuple)
		M,     # linear projector
		D,     # denoiser
		σ_0,   # starting sigma
		σ_L,   # last sigma
		h_0,   # starting h
		β,     # neg-temperature (set to 1 for pure hallucination)
		n      # hard limit for the number of iterations
		):
	σ = σ_0
	t = 1
	y = gaussian_image_drawer(s, 0.5, σ ** 2, t-1)
	while σ > σ_L and t < n:
		h = h_0 * t / (1 + h_0 * (t - 1))
		d = D(y) - y
		σ = (d.flatten().T @ d.flatten() / d.size)**0.5;
		γ = ((1 - β*h)**2 - (1 - h)**2)**0.5 * σ
		z = gaussian_image_drawer(s, 0, 1, t)
		y = y + h*d + γ*z
		print(f"t={t} σ={σ} h={h} γ={γ}")
		t = t + 1
	return y

# sample denoiser
def denoiser_median49(x):
	# <x morsi disk4.9 median >y
	# todo: rewrite with pipes
	import tempfile, iio, os
	X = f"{tempfile.NamedTemporaryFile().name}.npy"
	Y = f"{tempfile.NamedTemporaryFile().name}.npy"
	C = f"morsi disk4.9 median {X} {Y}"
	iio.write(X, x)
	os.system(C)
	y = iio.read(Y)
	os.unlink(X)
	os.unlink(Y)
	return y

# higher order function to create normalized denoisers
# input: a denoiser for images in the range [0,255]
# output: a denoiser for 0-centered images
# parameter: s = scale in 8-bit units
def normalized_denoiser(d, s=42):
	def f(x):
		y = 127 + s*x
		X = (d(y) - 127)/s
		return X
	return f

# same as above, but with a sigma
def normalized_denoiser_with_sigma(d, s=42, σ=10):
	def f(x):
		y = 127 + s*x
		X = (d(y, sigma=σ) - 127)/s
		return X
	return f


def qauto(x):
	y = x.flatten().copy()
	y.sort()
	n = y.size
	m = y[1*n//100]
	M = y[99*n//100]
	return 255 * (x - m) / (M - m)


# extract a named option from the command line arguments
def pick_option(
		o,  # option name, including hyphens
		d   # default value
		):
	from sys import argv as v
	return type(d)(v[v.index(o)+1]) if o in v else d


# main function
if __name__ == "__main__":
	d  = pick_option("-D", "median49")  # denoiser
	w  = pick_option("-w", 256)         # output image width
	h  = pick_option("-h", 256)         # output image height
	β  = pick_option("-b", 1.0)         # beta (1 - temperature)
	σ0 = pick_option("-s0", 1.0)        # first sigma
	σL = pick_option("-sL", 0.001)      # last sigma
	n  = pick_option("-n", 1000)        # iteration limit
	h0 = pick_option("-h0", 0.1)        # first h
	σ  = pick_option("-s", 10)          # base denoiser sigma
	s  = pick_option("-S", 42)          # base denoiser scale normalization
	o  = pick_option("-o", "out.npy")   # output filename

	print(f"w={w} h={h} β={β} D={d}")

	D = denoiser_median49
	if d != "median49":
		import ipol
		ipol.DEBUG_LEVEL = 0
		i = getattr(ipol, d)
		D = normalized_denoiser_with_sigma(i, s, σ)
	x = kadkhodaie_simoncelli_algorithm_1((h,w), D, σ0, σL, h0, β, n)

	import iio
	#y = qauto(x)
	#iio.write(o, y)
	iio.write(o, x)
