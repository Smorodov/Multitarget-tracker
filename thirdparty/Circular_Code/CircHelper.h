// ==========================================================================
// Copyright (C) 2011 Lior Kogan (koganlior1@gmail.com)
// ==========================================================================

#pragma once

// ==========================================================================
static const double _2Pi= 6.2831853071795864769252867665590057683943387987502116419498891846156328125724179972560696;

// ==========================================================================
// square (x*x)
template <typename T>
T Sqr(const T& x)
{
    return x*x;
}

// ==========================================================================
// Floating-point modulo
// The result (the remainder) has the same sign as the divisor.
// Similar to matlab's mod(); Not similar to fmod() -   Mod(-3,4)= 1   fmod(-3,4)= -3
template<typename T>
T Mod(T x, T y)
{
    static_assert(!std::numeric_limits<T>::is_exact , "Mod: floating-point type expected");

    if (0. == y)
        return x;

    double m= x - y * floor(x/y);

    // handle boundary cases resulting from floating-point limited accuracy:

    if (y > 0)              // modulo range: [0..y)
    {
        if (m>=y)           // Mod(-1e-16             , 360.    ): m= 360.
            return 0;

        if (m<0 )
        {
            if (y+m == y)
                return 0  ; // just in case...
            else
                return y+m; // Mod(106.81415022205296 , _TWO_PI ): m= -1.421e-14
        }
    }
    else                    // modulo range: (y..0]
    {
        if (m<=y)           // Mod(1e-16              , -360.   ): m= -360.
            return 0;

        if (m>0 )
        {
            if (y+m == y)
                return 0  ; // just in case...
            else
                return y+m; // Mod(-106.81415022205296, -_TWO_PI): m= 1.421e-14
        }
    }

    return m;
}
