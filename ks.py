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


def estimate_residual_noise(x):
	from numpy import sqrt
	from numpy.linalg import norm
	return norm(x) / sqrt(x.shape[0] * x.shape[1])


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
	while σ > σ_L and t < 100:
		h = h_0 * t / (1 + h_0 * (t - 1))
		d = D(y) - y
		σ = estimate_residual_noise(d)
		γ = ((1 - β*h)**2 - (1 - h)**2)**0.5 * σ
		z = gaussian_image_drawer(s, 0, 1, t)
		y = y + h*d + γ*z
		print(f"t={t} σ={σ} h={h} γ={γ}")
		t = t + 1
	return y

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
