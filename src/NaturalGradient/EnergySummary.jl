struct EnergySummary{T <: Number}
    data::Vector{T}
    mean::T
    std_of_mean::Float64
    var::Float64
    std_of_var::Float64
    importance_weights::Union{Vector{Float64}, Nothing}
    saved_properties

    EnergySummary(data::Vector{T}, mean::T, std_of_mean::Float64, var::Float64, std_of_var::Float64, importance_weights::Union{Vector{Float64}, Nothing}; saved_properties=nothing) where T <: Number =
        new{T}(data, mean, std_of_mean, var, std_of_var, importance_weights, saved_properties)
end

EnergySummary(ψ::MPS, H::MPO; sample_nr=1000) = EnergySummary([Ek(ψ, H) for _ in 1:sample_nr])

function EnergySummary(Eks::Vector{Complex{Float64}}; importance_weights=nothing, mean_=nothing, var_=nothing, kwargs...)
    if any(imag.(Eks) .> 1e-10)
        if mean_ === nothing || var_ === nothing
            mean_, var_ = wmean_and_var(Eks; weights_=importance_weights)
        end
        Eks_c = Eks .- mean_

        local std_of_mean
        local std_of_var

        if importance_weights !== nothing
            # The estimator for the mean energy is defined as X_k = E_k * w_k; therefore, the variance of X_k represents the error of the estimator.
            std_of_mean = std(real.(Eks_c .* importance_weights))
            
            # The estimator for the variance of the energy is defined as X_k = (E_k - <E>)^2 * w_k; similarly, the variance of X_k represents the error of this estimator.
            std_of_var = std(real.((Eks_c .* conj(Eks_c)) .* importance_weights))
            
            # The Eks are multiplied by the square root of the importance weights. This ensures that the product with (Oks * sqrt(importance_weights)) correctly recovers the importance weights to the first power.
            Eks_c = Eks_c .* sqrt.(importance_weights)
        else
            std_of_mean = sqrt(real.(var_))
            std_of_var = std(Eks_c .* conj(Eks_c))
        end
        return EnergySummary(Eks_c, mean_, std_of_mean, real.(var_), real.(std_of_var), importance_weights; kwargs...)
    end
    return EnergySummary(real.(Eks); importance_weights, kwargs...)
end

function EnergySummary(Eks::Vector{Float64}; importance_weights=nothing, mean_=nothing, var_=nothing, kwargs...)
    if mean_ === nothing || var_ === nothing
        mean_, var_ = wmean_and_var(Eks; weights_=importance_weights)
    end
    Eks_c = real.(Eks .- mean_)

    local std_of_mean
    local std_of_var

    if importance_weights !== nothing
        # The estimator for the mean energy is defined as X_k = E_k * w_k; therefore, the variance of X_k represents the error of the estimator.
        std_of_mean = std(real.(Eks_c .* importance_weights))
        
        # The estimator for the variance of the energy is defined as X_k = (E_k - <E>)^2 * w_k; similarly, the variance of X_k represents the error of this estimator.
        std_of_var = std(real.(Eks_c .^2 .* importance_weights))
        
        # The Eks are multiplied by the square root of the importance weights. This ensures that the product with (Oks * sqrt(importance_weights)) correctly recovers the importance weights to the first power.
        Eks_c = Eks_c .* sqrt.(importance_weights)
    else
        std_of_mean = sqrt(real.(var_))
        std_of_var = std(Eks_c .^ 2)
    end

    return EnergySummary(Eks_c, mean_, std_of_mean, var_, std_of_var, importance_weights; kwargs...)
end

Statistics.mean(Es::EnergySummary) = Es.mean
Statistics.var(Es::EnergySummary) = Es.var
Statistics.std(Es::EnergySummary) = sqrt(Es.var)
Base.length(Es::EnergySummary) = length(Es.data)

function get_importance_weights(Es::EnergySummary)
    if Es.importance_weights === nothing
        return ones(length(Es))
    else
        return Es.importance_weights
    end
end

energy_error(Es::EnergySummary) = Es.std_of_mean / sqrt(length(Es))
energy_var_error(Es::EnergySummary) = Es.std_of_var / sqrt(length(Es))

function centered(Es::EnergySummary; mode=:importance_sqrt)
    if Es.importance_weights === nothing
        return Es.data
    end
    if mode == :importance_sqrt
        return Es.data
    elseif mode == :importance
        return Es.data .* sqrt.(Es.importance_weights)
    elseif mode == :no_importance
        return Es.data ./ sqrt.(Es.importance_weights)
    else
        error("mode should be :importance_sqrt, :importance or :no_importance. $mode was given.")
    end
end
function uncentered(Es::EnergySummary)
    Esd = centered(Es; mode=:no_importance)
    return Esd .+ Es.mean
end

function effective_sample_nr(Es::EnergySummary)
    if Es.importance_weights === nothing
        return length(Es)
    else
        error = energy_error(Es)
        return Es.var / error^2
    end
end

function Base.show(io::IO, Es::EnergySummary)
    error = energy_error(Es)
    digits = Int(min(ceil(-log10(error)), 10)) + 1
    E_str = "E = $(round(real(Es.mean), digits=digits)) ± $(round(error, digits=digits))"

    error2 = energy_var_error(Es)
    digits = Int(min(ceil(-log10(error2)), 10)) + 1
    Evar_str = "var(E) = $(round(Es.var, digits=digits)) ± $(round(error2, digits=digits))"
    
    N_eff = effective_sample_nr(Es)
    N_eff_str = ""
    if N_eff != length(Es)
        N_eff_str = ", Nₑ=$(isnan(N_eff) ? "NaN" : round(Int, N_eff))"
    end
    print(io, "EnergySummary($E_str, $Evar_str, Nₛ=$(length(Es))$N_eff_str)")
end


function EnergySummary(θ::ParameterTypes, Eks_; sample_nr=100, timer=TimerOutput(), kwargs_Eks=Dict(), kwargs...)
    out = @timeit timer "Eks" Eks_(θ, sample_nr; kwargs_Eks...)
    kwargs = Dict{Any, Any}(kwargs...)
    saved_properties = Dict{Symbol, Any}()
    local Eks
    if haskey(out, :Eks)
        Eks = out[:Eks]
    else error("Oks_and_Eks should return Dict with key :Eks") end
    if haskey(out, :Oks)
        warning("Oks should not be returned by the Eks function and will be ignored.")
    end
    if haskey(out, :weights)
        kwargs[:importance_weights] = out[:weights]
    end

    if Eks isa Tuple
        @assert length(Eks) == 3 "Eks should be a Tuple with 3 elements, Eks, Ek_mean and Ek_var"
        Eks, kwargs[:Eks_mean], kwargs[:Eks_var] = Eks
    end

    saved_properties = Dict{Symbol, Any}()
    for key in keys(out)
        if !(key in [:Eks, :Oks, :logψs, :samples, :weights])
            saved_properties[key] = out[key]
        end
    end
    kwargs[:saved_properties] = saved_properties

    return EnergySummary(Eks; kwargs...)
end
