using Arpack
using TRS

function circle_line_intersections(a1::T, a2::T, b::T, r::T) where T
    """
    Calculates the points intersecting points between the 2d plane defined
    as a1*x + a2*y = b
    and the circle x^2 + y^2 = r^2
    Returns x1, y1, x2, y2 such that [x1; y1] and [x2; y2] are the two intersecting points.
    If no intersection(s) exist then NaN values are returned.
    Reference: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """
    # Normalize, i.e. make [a1; a2] unit norm
    norm_a = sqrt(a1^2 + a2^2)
    if norm_a < 1e-11  # Very badly scaled problem
        @warn "Badly scaled circe-line"
        return T(NaN), T(NaN), T(NaN), T(NaN)
    end
    a1 /= norm_a; a2 /= norm_a; b /= norm_a

    # First calculate a point (x, y) on the line
    # and dx, dy such that (x + dx, y + dy) lies also on the line.
    if abs(a1) > abs(a2) # Avoid dividing with something small
        x = (b - a2)/a1
        y = one(T)

        dx = -a2/a1
        dy = one(T)
    else
        x = one(T)
        y = (b - a1)/a2

        dx = one(T)
        dy = -a1/a2
    end

    # Make (dx, dy) unit norm
    norm_d = sqrt(dx^2 + dy^2)
    dx /= norm_d; dy /= norm_d

    D = x*(y + dy) - (x + dx)*y
    #= Allow for slightly infeasible solutions
    if abs(r)/abs(D) <= one(T)
        @show abs(r)/abs(D) - 1
        if abs(r)/abs(D) >= one(T) - 1e-11
            D = r
        end
    end
    =#
    if r >= abs(D)
        c = sqrt(r^2 - D^2)

        x1 = D*dy; y1 = -D*dx
        x2 = x1;  y2 = y1;

        δ1 = dx*c
        if dy < 0; δ1 = -δ1 end
        δ2 = abs(dy)*c

        x1 += δ1; y1 += δ2
        x2 -= δ1; y2 -= δ2

        return x1, y1, x2, y2
    else
        return T(NaN), T(NaN), T(NaN), T(NaN)
    end
end

function trs_robust(P::AbstractArray{T}, q::AbstractVector{T}, r::T; tol=0.0, kwargs...) where T
    n = length(q)
    if n < 15
        return trs_boundary_small(P, q, r; kwargs...)
    else
        try
            return trs_boundary(P, q, r; tol=tol, kwargs...)
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