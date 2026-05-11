function wmean(arr::AbstractArray{<:Number}; weights_=nothing, kwargs...)
    # TODO: Parallelize this function
    if weights_ !== nothing
        return mean(arr, weights(weights_); kwargs...)
    end
    return mean(arr; kwargs...)
end
function get_sum_size(arr::AbstractArray{<:Number}; dims=nothing, kwargs...)
    if dims === nothing
        return prod(size(arr))
    else
        if dims isa Int
            dims = [dims]
        end
        return prod(size(arr)[i] for i in dims)
    end
end

function wmean_and_var(arr::AbstractArray{<:Number}; weights_=nothing, kwargs...)
    local mean_, var_
    size_ = get_sum_size(arr; kwargs...)
    if weights_ !== nothing
        @assert prod(size(weights_)) == size_ "The size of the weights_ array must be the same as the dimensions of the array to be averaged."
        n = mean(weights_; kwargs...)
        mean_ = wmean(arr; weights_, kwargs...) ./ n
        N_eff = sum(weights_)^2/sum(weights_.^2)
        f = N_eff/(N_eff - 1)
        arr_m_ = arr .- mean_
        var_ = wmean(arr_m_ .* conj(arr_m_); weights_, kwargs...) * f

    else
        mean_ = mean(arr; kwargs...)
        f = size_ / (size_ - 1)
        arr_m_ = arr .- mean_
        var_ = mean(arr_m_ .* conj(arr_m_); kwargs...) .* f
    end
    return mean_, var_
end

wvar(arr::AbstractArray{<:Number}; kwargs...) = wmean_and_var(arr; kwargs...)[2]
wstd(arr::AbstractArray{<:Number}; kwargs...) = sqrt.(wvar(arr; kwargs...))

function threaded_mean(arr::Vector{T}; weights_=nothing) where T
    num_threads = Threads.nthreads()
    sums = Vector{T}(undef, num_threads)
    local norm
    len = length(arr)
    if weights_ === nothing
        @threads for i in 1:num_threads
            start_index = div(len*(i - 1), num_threads) + 1
            end_index = div(len*i, num_threads)
            sums[i] = sum(arr[start_index:end_index])
        end
        norm = len
    else
        @assert length(weights_) == len
        norms = Vector{T}(undef, num_threads)
        @threads for i in 1:num_threads
            start_index = div(len*(i - 1), num_threads) + 1
            end_index = div(len*i, num_threads)
            sums[i] = sum(arr[start_index:end_index].*weights_[start_index:end_index])
            norms[i] = sum(weights_[start_index:end_index])
        end
        norm = sum(norms)
    end

    # Compute the overall mean
    return sum(sums) ./ length(arr)
end

function convert_to_matrix_without_mean(m::Vector{Vector{T}}; mean_=nothing, kwargs...) where T <: Number
    if mean_ === nothing
        mean_ = threaded_mean(m; kwargs...)
    end

    l2 = length(m[1])
    M = Matrix{T}(undef, length(m), l2)
    
    Threads.@threads for i in 1:length(m)
        #@assert length(m[i]) == l2
        M[i, :] .= m[i] .- mean_
    end

    return M, mean_
end