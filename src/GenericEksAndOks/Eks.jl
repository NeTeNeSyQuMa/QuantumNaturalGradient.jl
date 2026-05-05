abstract type AbstractTensorOperatorSum end

struct TensorOperatorSum <: AbstractTensorOperatorSum
    tensors::Vector{ITensor}
    hilbert::Array{<:Index}
    sites::Vector{Vector}
end
Base.size(t::TensorOperatorSum, args...) = size(t.hilbert, args...)
Base.ndims(t::TensorOperatorSum) = ndims(t.hilbert)

Base.show(io::IO, t::TensorOperatorSum) = print(io, "TensorOperatorSum(nr_tensors=$(length(t.tensors)), hilbert_size=$(size(t)))")


function insert_JW_string(o::ITensors.Scaled, hilbert::Array)
    c = ITensors.coefficient(o)
    p = ITensors.argument(o)  # Prod{Op}
    return c * insert_JW_string(p, hilbert)
end

"""
    insert_JW_string(p::ITensors.Prod{ITensors.Op}, hilbert)

    Finds the left and right most site in `p` and inserts a JW string inbetween if the sitetype is fermionic.
"""
function insert_JW_string(p::ITensors.Prod{ITensors.Op}, hilbert::Array)
    ops = collect(p)
    fermionic_sites = [only(ITensors.sites(op)) for op in ops
                       if ITensors.has_fermion_string(ITensors.which_op(op), hilbert[only(ITensors.sites(op))])]
    
    # If there are no fermionic sites, return the original operator
    # Note: N=Cdag C also has no fermionic string
    isempty(fermionic_sites) && return p

    # find left and right most site in the ordering
    l = minimum(fermionic_sites) # left most site
    r = maximum(fermionic_sites) # right most site

    # insert JW string between l and r
    for n in (l+1):(r-1)
        p *= ITensors.Op("F", n)
    end
    return p
end

"""
    TensorOperatorSum(tensors, hilbert, sites)
    Generates a TensorOperatorSum from a hamiltonian and a hilbert space. It precomputes the sites where the operator acts on.
"""
function TensorOperatorSum(ham::OpSum, hilbert::Array; combine_tensors=true)
    
    tensors = Vector{ITensor}(undef, length(ham))
    sites = Vector{Vector{Int}}(undef, length(ham))
    hilbert_flat = hilbert
    if ndims(hilbert) > 1
        hilbert_flat = hilbert[:]
        ## Reduce dimensionality of hamiltonian to 1D for compatibility pourposes
        ham = reduce_dim(ham, size(hilbert))
    end
    
    isfermionic = hastags(hilbert_flat[1], "Fermion")

    for (i, o) in enumerate(ham)
        o = isfermionic ? insert_JW_string(o, hilbert_flat) : o
        tensors[i] = ITensor(o, hilbert[:])
        sites[i] = get_active_sites(o)
    end
    tso = TensorOperatorSum(tensors, hilbert, sites)
    if combine_tensors
        tso = combine_tensors_at_same_site(tso)
    end

    cast_real_if_complex_is_zero!(tso)
    return tso
end

convert_eltype(::Type{S}, t::ITensor) where S = ITensor(S.(t.tensor))
TensorOperatorSum(::Type{S}, t::TensorOperatorSum) where S = TensorOperatorSum(convert_eltype.(S, t.tensors), t.hilbert, t.sites)


function get_precomp_sOψ_elems_slow!(tensor::ITensor, sites::Vector, sample_, hilbert; sum_precompute=DefaultOrderedDict(()->0), offset=1, get_flip_sites=false)
    sample_r = sample_[sites]
    hilbert_r = hilbert[sites]
    
    indices_sample = collect(hi' => s for (hi, s) in zip(hilbert_r, sample_r)) # Selects the indices that act on the tensor from the left O|s>
    
    for sample_r2 in bitstring_permutations(length(sites))
        sample_r2 .+= offset
        indices = vcat(indices_sample, collect(hi => s for (hi, s) in zip(hilbert_r, sample_r2))) # <s'|
        vi = tensor[indices...] # <s'|O|s>
        if vi != 0.
            key = find_flip_site(sample_r .- offset, sample_r2 .- offset, sites)
            sum_precompute[key] += vi
        end
    end

    return sum_precompute
end

"""
Generates a Dictionary with the entries being (sample_config => energy_weight) so that Ek=sum energy_weight * exp(logψ(sample_config) - logψ(sample))
get_precomp_sOψ_elems(
    tensor::ITensor, 
    sites::Vector, 
    sample_, 
    hilbert; 
    sum_precompute = DefaultOrderedDict(()->0), 
    offset=1, 
    get_flip_sites=false # If true, it will give a Dictionary Dict(sites_flipped=>energy_weight) instead of Dict(s'=>energy_weight)
    sites_fliiped: First index is the site, second the new s'_i.
)
"""
function get_precomp_sOψ_elems!(tensor::ITensor, sites::Vector, sample_, hilbert; sum_precompute=DefaultOrderedDict(()->0), offset=1)
    
    sample_r = sample_[sites]
    hilbert_r = hilbert[sites]
    
    indices_sample = collect(hi' => s for (hi, s) in zip(hilbert_r, sample_r)) # Selects the indices that act on the tensor from the left O|s>
    
    tensor_proj = onehot(eltype(tensor), indices_sample) * tensor # <s'|T
    
    # Make sure that the indices have the right permutation
    perm = NDTensors.getperm(ITensors.inds(tensor_proj), hilbert_r)
    tensor_proj = ITensor(permutedims(tensor_proj.tensor, perm))
    
    inds = findall(x -> x != 0, tensor_proj.tensor)
    for ind in inds
        if length(ind) == 1
            sample_r2 = ind
        else
            sample_r2 = ind.I
        end
        vi = tensor_proj[ind]
        key = find_flip_site(sample_r .- offset, sample_r2 .- offset, sites)
        # Key is in the format ((site_flipped, s'_i), ...)
        sum_precompute[key] += vi
    end
    
    return sum_precompute
end

"""
find_flip_site([1, 2], [2,2], [4, 5])

1-element Vector{Any}:
 (4, 2)
"""
function find_flip_site(sample_orig, sample_shifted, sites)
    diffs = []
    for (i, s_orig, s_shift) in zip(sites, sample_orig, sample_shifted)
        if s_orig != s_shift
            push!(diffs, (i, s_shift))
        end
    end
    return (diffs...,)
end

function apply_flip_site(sample, patch)
    sample_c = copy(sample)
    for (index, value) in patch
        sample_c[index...] = value
    end
    return sample_c
end

function get_precomp_sOψ_elems(tso::Vector{TensorOperatorSum}, sample_::Array; sum_precompute = DefaultOrderedDict(()->0), offset=1, get_flip_sites=false)
    for tso_i in tso
        sum_precompute = get_precomp_sOψ_elems(tso_i, sample_; sum_precompute, offset, get_flip_sites)
    end
    return sum_precompute
end


function get_precomp_sOψ_elems(tso::TensorOperatorSum, sample_::Array{T, N}; sum_precompute=DefaultOrderedDict(()->0), offset=1, get_flip_sites=false, kwargs...) where {T <: Int, N}
    @assert size(sample_) == size(tso)
    sample_ = sample_[:] # Flatten the sample
    sample_o = sample_ .+ offset # Shift the sample from 0...N-1 to 1...N
    @assert all(sample_o .> 0) "Sample must be composed of positive integers instead of $sample_"
    
    for (tensor, sites) in zip(tso.tensors, tso.sites)
        get_precomp_sOψ_elems!(tensor, sites, sample_o, tso.hilbert[:]; sum_precompute, offset, kwargs...)
    end
    
    # Remove zeros and make real if the imaginary part is too small
    for (key, value) in sum_precompute
        if value == 0.
            delete!(sum_precompute, key)
        elseif value isa Complex && abs(imag(value))/(abs(real(value))+1e-10) < 1e-14
            sum_precompute[key] = real(value)
        end
    end

    # If get_flip_sites is false it will compute the samples with the flipped sites applied to it
    if !get_flip_sites
        for (diff, v) in copy(sum_precompute) # Copy to avoid changing the dictionary while iterating
            if diff isa Tuple
                sample_flipped = apply_flip_site(sample_, diff)
                sum_precompute[sample_flipped] += v
                delete!(sum_precompute, diff)
            end
        end
    end
    
    # If not 1D, reshape the samples
    if ndims(tso) > 1
        increase_dim!(sum_precompute, size(tso))
    end
    return sum_precompute
end

function inner_div_ψ(sample_, tso::TensorOperatorSum, logψ_func; logψ_sample=nothing)
    sum_precompute = get_precomp_sOψ_elems(tso, sample_)
    
    if logψ_sample === nothing
        logψ_sample = logψ_func(sample_)
    end
    
    O0 = 0
    if sample_ in keys(sum_precompute)
        O0 = sum_precompute[sample_]
        delete!(sum_precompute, sample_)
    end
    
    O = sum(weight * exp(logψ_func(sample_r) - logψ_sample) for (sample_r, weight) in sum_precompute; init=O0)
    if O isa Complex && abs(imag(O))/(abs(real(O))+1e-10) < 1e-14
        O = real(O)
    end
    return O
end

get_Ek(sample_, tso::TensorOperatorSum, logψ_func; logψ_sample=nothing) = inner_div_ψ(sample_, tso, logψ_func; logψ_sample)

inner_div_ψ(sample_, opsum::OpSum, logψ_func, hilbert; logψ_sample=nothing) = inner_div_ψ(sample_, TensorOperatorSum(opsum, hilbert), logψ_func; logψ_sample)
get_Ek(sample_, opsum::OpSum, logψ_func, hilbert; logψ_sample=nothing) = inner_div_ψ(sample_, opsum, logψ_func, hilbert; logψ_sample)

# TODO write a more general version for higher physical dimensions
function bitstring_permutations(n)
    ss = []
    for i = 0:2^n-1
       s = bitstring(i)
       s = s[end-n+1:end]
       s = [parse(Int, si) for si in s]
       push!(ss, s)
    end
    return ss
end

function get_active_sites(o::Scaled)
    return [a.sites[1] for a in ITensors.argument(o).args[1]]
end

## Reduce dimensionality of hamiltonian to 1D for compatibility pourposes
reduce_dim(O::Op, size::Tuple) = ITensors.Op(O.which_op, reduce_dim(O.sites[1], size))
reduce_dim(v::Vector, size::Tuple) = [reduce_dim(vi, size) for vi in v]
function reduce_dim(t::Tuple, size::Tuple)
    n = 0
    mul = 1
    for (ti, si) in zip(t, size)
        n += mul * (ti-1)
        mul *= si
    end
    return n + 1
end
function increase_dim(t::Integer, size::Tuple)
    n = t - 1
    t = []
    for si in size
        push!(t, n % si + 1)
        n ÷= si
    end
    return tuple(t...)
end

function increase_dim(ts::Tuple, size::Tuple)
    if length(ts) == length(size)
        return ts
    elseif length(ts) == 1
        return increase_dim(ts[1], size)
    else
        error("Not Implemented")
    end
end

function increase_dim(sample_::AbstractArray, size_)
    sample_high = reshape(sample_, size_)
    @assert size(sample_high) != size(sample_) "The size of the sample should be different from the original size"
    return sample_high
end


function increase_dim(sum_precompute::DefaultOrderedDict, size_)
    sum_precompute2 = DefaultOrderedDict(()->0)
    for (obj, v) in sum_precompute
        if obj isa Tuple # If it is a tuple, it means that it stores the difference between the original S and the flipped S'
            diff = obj
            diff_res = []
            for (i, s) in diff
                push!(diff_res, (increase_dim(i, size_), s))
            end
            diff_res = Tuple(diff_res)
            sum_precompute2[diff_res] += v
        
        elseif obj isa AbstractArray # If it is not a tuple, it means that it should stores S'
            sample__ = obj
            sum_precompute2[reshape(sample__, size_)] += v
        
        else
            error("sum_precompute should be composed of Tuples or AbstractArrays")
        end
    end
    return sum_precompute2
end

function increase_dim!(sum_precompute::DefaultOrderedDict, size_)
    for (obj, v) in copy(sum_precompute)
        if obj isa Tuple # If it is a tuple, it means that it stores the difference between the original S and the flipped S'
            diff = obj
            diff_high = []
            for (i, s) in diff
                push!(diff_high, (increase_dim(i, size_), s))
            end
            diff_high = Tuple(diff_high)
            if diff_high != diff
                sum_precompute[diff_high] += v
                delete!(sum_precompute, obj)
            end
        
        elseif obj isa AbstractArray # If it is not a tuple, it means that it should stores S'
            sample__ = obj
            sample_high = reshape(sample__, size_)
            @assert size(sample_high) != size(sample__) "The size of the sample should be different from the original size"
            sum_precompute[sample_high] += v
            delete!(sum_precompute, sample__)
        
        else
            error("sum_precompute should be composed of Tuples or AbstractArrays")
        end
    end
    return sum_precompute
end


function reduce_dim!(O::ITensors.Scaled, size::Tuple)
    a = ITensors.argument(O)
    a.args[1] .= reduce_dim(a.args[1], size)
end

function reduce_dim(O::ITensors.OpSum, size::Tuple)
    O_c = deepcopy(O)
    for oi in O_c
        reduce_dim!(oi, size)
    end
    return O_c
end

second(x) = x[2]
function combine_tensors_at_same_site(tso::TensorOperatorSum)
    d = Dict()
    for (sites, tensor) in zip(tso.sites, tso.tensors)
        if sites in keys(d)
            d[sites] .+= tensor 
        else
            d[sites] = tensor         
        end
    end
    tensors_sites = collect(d)
    sites = first.(tensors_sites)
    tensors = second.(tensors_sites)

    return TensorOperatorSum(tensors, tso.hilbert, sites)
end

function cast_real_if_complex_is_zero!(tso::TensorOperatorSum)
    for (i, t) in enumerate(tso.tensors)
        if eltype(t) <: Complex && norm(imag(t)) < 1e-14
            tso.tensors[i] = real(t)
        end
    end
end