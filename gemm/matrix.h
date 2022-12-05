#pragma once

#ifdef __CUDACC__
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif 

class matrix_properties
{
	size_t __size_x;
	size_t __size_y;
	size_t __size_z;
	size_t __size_l;

public:
	CUDA_CALLABLE_MEMBER matrix_properties(size_t width, size_t height, size_t depth, size_t layer) :
		__size_x(width), __size_y(height), __size_z(depth), __size_l(layer)
	{;}

	CUDA_CALLABLE_MEMBER matrix_properties(const matrix_properties&) = default;

	CUDA_CALLABLE_MEMBER size_t size_x() const noexcept
	{
		return __size_x;
	}

	CUDA_CALLABLE_MEMBER size_t size_y() const noexcept
	{
		return __size_y;
	}

	CUDA_CALLABLE_MEMBER size_t size_z() const noexcept
	{
		return __size_z;
	}

	CUDA_CALLABLE_MEMBER size_t size_l() const noexcept
	{
		return __size_l;
	}

	CUDA_CALLABLE_MEMBER size_t length() const noexcept
	{
		return __size_x * __size_y * __size_z * __size_l;
	}
	
	CUDA_CALLABLE_MEMBER size_t index(size_t x, size_t y, size_t z, size_t l) const noexcept
	{
		size_t lspec, index;

		lspec = 1U;
		index = lspec * (x % size_x());

		lspec *= size_x();
		index += lspec * (y % size_y());

		lspec *= size_y();
		index += lspec * (z % size_z());

		lspec *= size_z();
		index += lspec * (l % size_l());

		return index;
	}
};

template<class T>
class raw_memory_matrix_view
{
	matrix_properties __accessor;

protected:
	T*                __data{ nullptr };

public:
	CUDA_CALLABLE_MEMBER raw_memory_matrix_view() :
		__accessor{ 0, 0, 0, 0 }, __data{ nullptr }
	{;}

	CUDA_CALLABLE_MEMBER raw_memory_matrix_view(raw_memory_matrix_view<T>& other) :
		__accessor { other.__accessor }, __data { other.__data }
	{;}

	CUDA_CALLABLE_MEMBER raw_memory_matrix_view(size_t width, size_t height, size_t depth, size_t count_of_layers) :
		__accessor{ width, height, depth, count_of_layers }, __data{ nullptr }
	{;}

public:
	CUDA_CALLABLE_MEMBER raw_memory_matrix_view(T* data, size_t width, size_t height, size_t depth, size_t count_of_layers) :
		__accessor{ width, height, depth, count_of_layers }, __data{ data }
	{
	}

	CUDA_CALLABLE_MEMBER raw_memory_matrix_view(T* data, const matrix_properties& props) :
		__accessor{ props }, __data{ data }
	{
	}

	template<class Tlambda>
	CUDA_CALLABLE_MEMBER void for_each(Tlambda&& for_cell) const
	{
		auto size_x = __accessor.size_x();
		auto size_y = __accessor.size_y();
		auto size_z = __accessor.size_z();
		auto size_l = __accessor.size_l();

		for (size_t l = 0; l < size_l; ++l)
		{
			for (size_t z = 0; z < size_z; ++z)
			{
				for (size_t y = 0; y < size_y; ++y)
				{
					for (size_t x = 0; x < size_x; ++x)
					{
						for_cell(x, y, z, l);
					}
				}
			}
		}
	}

	CUDA_CALLABLE_MEMBER inline T* data()
	{
		return __data;
	}

	CUDA_CALLABLE_MEMBER inline T& at(size_t x, size_t y, size_t z, size_t l)
	{
		return __data[__accessor.index(x, y, z, l)];
	}

	CUDA_CALLABLE_MEMBER inline T& at(size_t x, size_t y, size_t z, size_t l) const
	{
		return __data[__accessor.index(x, y, z, l)];
	}

	CUDA_CALLABLE_MEMBER inline T& at(size_t x, size_t y)
	{
		return __data[__accessor.index(x, y, 0, 0)];
	}

	CUDA_CALLABLE_MEMBER inline T& at(size_t x, size_t y) const
	{
		return __data[__accessor.index(x, y, 0, 0)];
	}

	CUDA_CALLABLE_MEMBER const matrix_properties& properties() const
	{
		return __accessor;
	}

	CUDA_CALLABLE_MEMBER size_t size_of_data() const noexcept
	{
		return __accessor.length() * sizeof(T);
	}
};
