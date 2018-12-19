using GeneralQP

mutable struct NullspaceHessian{T}
    n::Int # Dimension of the original space
    m::Int # Dimension of the nullspace

    P::Matrix{T}     # The full hessian
    # Z is the nullspace, i.e. QR.Q2 where QR is defined below
    Z::SubArray{T, 2, Matrix{T}, Tuple{Base.Slice{Base.OneTo{Int}}, UnitRange{Int}}, true}
    ZPZ::SubArray{T, 2, Matrix{T}, Tuple{UnitRange{Int},UnitRange{Int}}, false}  # equal to Z'*P*Z
    QR::UpdatableQR{T}
    data::Matrix{T}  # That's where ZPZ is viewing into

    function NullspaceHessian{T}(P, A) where {T}
        @assert(size(A, 1) == size(P, 1) == size(P, 2), "Matrix dimensions do not match.")

        F = UpdatableQR(A)
        n = F.n
        m = F.n - F.m

        data = zeros(T, n, n)
        ZPZ = view(data, n-m+1:n, n-m+1:n) 
        ZPZ .= F.Q2'*P*F.Q2; ZPZ .= (ZPZ .+ ZPZ')./2

        new{T}(n, m, P, F.Q2, ZPZ, F, data)
    end
end

function add_constraint!(H::NullspaceHessian{T}, a::Vector{T}) where {T}
    a2 = add_column!(H.QR, a)

    for i = length(a2):-1:2
        G, r = givens(a2[i-1], a2[i], i-1, i)
        lmul!(G, a2)
        rmul!(H.ZPZ, G')
        lmul!(G, H.ZPZ)
    end
    # ToDo: Force symmetry? (i.e. H.ZPZ .= (H.ZPZ .+ H.ZPZ')./2)
    H.m -= 1; update_views!(H)

    return nothing
end

function remove_constraint!(H::NullspaceHessian{T}, idx::Int) where{T}
    remove_column!(H.QR, idx)
    H.m += 1; update_views!(H)

    Pz = H.P*view(H.Z, :, 1)  # ToDo: avoid memory allocation
    mul!(view(H.ZPZ, 1, :), H.Z', Pz)
    for i = 2:H.m
        H.ZPZ[i, 1] = H.ZPZ[1, i]
    end
    
    return nothing
end

function update_views!(H::NullspaceHessian{T}) where {T}
    range = H.n-H.m+1:H.n
    H.ZPZ = view(H.data, range, range)
    H.Z = H.QR.Q2
end

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
    if r >= D
        c = sqrt(r^2 - D^2)

        y11 = D*a1; y12 = D*a2
        y21 = y11;  y22 = y12;

        δ1 = -sign(a1)*a2*c
        δ2 = abs(a1)*c

        y11 += δ1; y12 += δ2
        y21 -= δ1; y22 -= δ2

        return y11, y12, y21, y22
    else
        return T(NaN), T(NaN), T(NaN), T(NaN)
    end
end