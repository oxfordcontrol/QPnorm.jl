using Arpack
using TRS

function circle_line_intersections(a1::T, a2::T, b::T, r::T) where T
    """
    Calculates the points intersecting points between the 2d plane defined
    as a1*y1 + a2*y2 <= b
    and the circle y1^2 + y^2 = r^2

    Returns y11, y12, y21, y22 such that [y11; y12] and [y21; y22] are the two intersecting points.
    If no intersection(s) exist then NaN values are returned.

    Reference: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """
    inv_norm_a = one(T)/sqrt(a1^2 + a2^2)
    if inv_norm_a > 1e15  # Very badly scaled problem or LICQ is violated
        return T(NaN), T(NaN), T(NaN), T(NaN)
    end
    a1 *= inv_norm_a; a2 *= inv_norm_a; b *= inv_norm_a

    x1 = a1*b; x2 = a2*b
    # [x1; x2 + a1] and [x2; x1 - a2] are two points in the plane

    D = x1*(x2 + a1) - x2*(x1 - a2)
    if r >= abs(D)
        c = sqrt(r^2 - D^2)

        y11 = D*a1; y12 = D*a2
        y21 = y11;  y22 = y12;

        δ1 = -sign(a1)*a2*c
        δ2 = abs(a1)*c

        y11 += δ1; y12 += δ2
        y21 -= δ1; y22 -= δ2

        # @show maximum(a1*y11 + a2*y12 - b), abs(norm([y11; y12]) - r)
        return y11, y12, y21, y22
    else
        return T(NaN), T(NaN), T(NaN), T(NaN)
    end
end

function trs_robust(P::AbstractArray{T}, q::AbstractVector{T}, r::T; kwargs...) where T
    n = length(q)
    if n < 15
        return trs_boundary_small(P, q, r; kwargs...)
    else
        # return trs_boundary(P, q, r; kwargs...)
        try
            x_g, x_l, info = trs_boundary(P, q, r; kwargs...)
            #=
            x_g1, x_l1, info1 = trs_boundary_small(P, q, r; kwargs...)
            @show info
            @show info1
            @show norm(x_g - x_g1)
            =#
            return x_g, x_l, info
        catch e
            if isa(e, LAPACKException)
                try
                    @warn "Error #1:", typeof(e), " - trying again"
                    return trs_boundary(P, q, r; kwargs...)
                catch
                    #    @show e
                    @warn "Error #2:", e, " - switching to direct"
                    return trs_boundary_small(P, q, r; kwargs...)
                end
            else
                @warn "Error:", e, " - switching to direct"
                return trs_boundary_small(P, q, r; kwargs...)
            end
        end
    end
end