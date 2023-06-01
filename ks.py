# draw a (deterministic) gaussian noise image
def gaussian_image_drawer(
		s,     # shape tuple
		μ,     # mean
		σ,     # variance
		r      # random seed
		):
	import os, iio
	os.environ["SRAND"] = f"{r}"
	x = iio.read(f"TRANS[pipe=plambda randg]:zero:{s[0]}x{s[1]}")
	return μ + σ*x


# Kadkhodaie Simoncelli dynamics
# "Solving Linear Inverse Problems Using the Prior Implicit in a Denoiser"
# Implementation of Algorithm 1 from https://arxiv.org/abs/2007.13640
def kadkhodaie_simoncelli(
		s,     # image shape (2 or 3-tuple)
		D,     # denoiser
		σ_0,   # starting sigma
		σ_L,   # last sigma
		h_0,   # starting h
		β      # neg-temperature (set to 1 for pure hallucination)
		):
	σ = σ_0
	t = 1
	y = gaussian_image_drawer(s, 0.5, σ ** 2, t-1)
	while σ > σ_L and t < 150:
		h = h_0 * t / (1 + h_0 * (t - 1))
		d = D(y) - y
		σ = (y.flatten().T @ y.flatten() / y.size)**0.5;
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


def qauto(x):
	y = x.flatten().copy()
	y.sort()
	n = y.size
	m = y[5*n//100]
	M = y[95*n//100]
	return 255 * (x - m) / (M - m)


# extract a named option from the command line arguments (sys.argv is edited)
def pick_option(
		o,  # option name, including hyphens
		d   # default value
		):
	from sys import argv as v
	return type(d)(v[v.index(o)+1]) if o in v else d

if __name__ == "__main__":
	import sys
	print(f"sys.argv={sys.argv}")


# main function
if __name__ == "__main__":
	d  = pick_option("-D", "median49")  # denoiser
	w  = pick_option("-w", 256)         # output image width
	h  = pick_option("-h", 256)         # output image height
	β  = pick_option("-b", 1.0)         # beta (1 - temperature)
	σ0 = pick_option("-s0", 1.0)        # first sigma
	σL = pick_option("-sL", 0.001)      # last sigma
	h0 = pick_option("-h0", 0.1)        # first h
	o  = pick_option("-o", "out.npy")   # output filename

	print(f"w={w} h={h} β={β} D={d}")

	D = denoiser_median49
	x = kadkhodaie_simoncelli((h,w), D, σ0, σL, h0, β)

	import iio
	y = qauto(x)
	iio.write(o, y)
