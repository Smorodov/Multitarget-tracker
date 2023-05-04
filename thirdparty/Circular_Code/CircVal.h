// ==========================================================================
// Copyright (C) 2011 Lior Kogan (koganlior1@gmail.com)
// ==========================================================================
// classes defined here:
// CircVal            - circular-value
// CircValTester      - tester for CircVal class
// ==========================================================================

#pragma once

//#define _USE_MATH_DEFINES // M_PI
#include <cmath>
#include <random>
#include <limits>
#include <assert.h>

#include "FPCompare.h"
#include "CircHelper.h"

// ==========================================================================
// macro for defining a circular-value type
#define CircValTypeDef(_Name, _L, _H, _Z)             \
    struct _Name                                      \
    {                                                 \
        static const double L  ; /* range: [L,H) */   \
        static const double H  ;                      \
        static const double Z  ; /* zero-value   */   \
        static const double R  ; /* range        */   \
        static const double R_2; /* half range   */   \
    };                                                \
                                                      \
    const double _Name::L  = (_L)          ;          \
    const double _Name::H  = (_H)          ;          \
    const double _Name::Z  = (_Z)          ;          \
    const double _Name::R  = ((_H)-(_L))   ;          \
    const double _Name::R_2= ((_H)-(_L))/2.;

// ==========================================================================
// basic circular-value types
CircValTypeDef(SignedDegRange  , -180.,   180.,  0. )
CircValTypeDef(UnsignedDegRange,    0.,   360.,  0. )
CircValTypeDef(SignedRadRange  , -M_PI,   M_PI,  0. )
CircValTypeDef(UnsignedRadRange,    0., 2*M_PI,  0. )

// some additional circular-value types - for testing
CircValTypeDef(TestRange0      ,    3.,    10.,  5.3)
CircValTypeDef(TestRange1      ,   -3.,    10., -3.0)
CircValTypeDef(TestRange2      ,   -3.,    10.,  9.9)
CircValTypeDef(TestRange3      ,  -13.,    -3., -5.3)


// ==========================================================================
// circular-value
// Type should be defined using the CircValTypeDef macro
template <typename Type>
class CircVal
{
    double val; // actual value [L,H)

    // ---------------------------------------------
public:
    static double GetL() { return Type::L; }
    static double GetH() { return Type::H; }
    static double GetZ() { return Type::Z; }
    static double GetR() { return Type::R; }

    // ---------------------------------------------
    static bool IsInRange(double r)
    {
        return (r>=Type::L && r<Type::H);
    }

    // 'wraps' circular-value to [L,H)
    static double Wrap(double r)
    {
        // the next lines are for optimization and improved accuracy only
        if (r>=Type::L)
        {
                 if (r< Type::H        ) return r        ;
            else if (r< Type::H+Type::R) return r-Type::R;
        }
        else
                 if (r>=Type::L-Type::R) return r+Type::R;

        // general case
        return Mod(r - Type::L, Type::R) + Type::L;
    }

    // ---------------------------------------------
    // the length of shortest directed walk from c1 to c2
    // return value is in [-R/2, R/2)
    static double Sdist(const CircVal& c1, const CircVal& c2)
    {
        double d= c2.val-c1.val;
        if (d <  -Type::R_2) return d + Type::R;
        if (d >=  Type::R_2) return d - Type::R;
                             return d          ;
    }

    // the length of the shortest increasing walk from c1 to c2
    // return value is in [0, R)
    static double Pdist(const CircVal& c1, const CircVal& c2)
    {
        return c2.val>=c1.val ? c2.val-c1.val : Type::R-c1.val+c2.val;
    }

    // ---------------------------------------------
    CircVal()
    {
        val= Type::Z;
    }

    // construction based on a floating-point value
    // should only be called when the floating-point is a value in the range!
    // to translate a floating-point such that 0 is mapped to Type::Z, call ToC()
    CircVal(double r)
    {
        val= Wrap(r);
    }

    // construction based on a circular value of the same type
    CircVal(const CircVal& c)
    {
        val= c.val;
    }

    // construction based on a circular value of another type
    // sample use: CircVal<SignedRadRange> c= c2;   -or-   CircVal<SignedRadRange> c(c2);
    template<typename CircVal2>
    CircVal(const CircVal2& c2)
    {
        double val2= Pdist(CircVal2::GetZ(), c2);
        val= Wrap(val2*Type::R / CircVal2::GetR() + Type::Z);
    }

    // ---------------------------------------------
    operator double() const
    {
        return val;
    }

    // ---------------------------------------------
    // assignment from a floating-point value
    // should only be called when the floating-point is a value in the range!
    // to translate a floating-point such that 0 is mapped to Type::Z, call ToC()
    CircVal& operator= (double r)
    {
        val= Wrap(r);
        return *this;
    }

    // assignment from another type of circular value
    template<typename CircVal2>
    CircVal& operator= (const CircVal2& c2)
    {
        double val2= c2.Pdist(c2.GetZ(), c2);
        val= Wrap(val2*Type::R/c2.GetR() + Type::Z);
        return *this;
    }

    // ---------------------------------------------
    // convert circular-value c to real-value [L-Z,H-Z). .Z is converted to 0
    friend double ToR(const CircVal& c) { return c.val - Type::Z; }

    // ---------------------------------------------
    const CircVal  operator+ (                ) const { return val;                                            }
    const CircVal  operator- (                ) const { return Wrap(Type::Z-Sdist(Type::Z,val));               } // return negative circular value
    const CircVal  operator~ (                ) const { return Wrap(val+Type::R_2             );               } // return opposite circular-value

    const CircVal  operator+ (const CircVal& c) const { return Wrap(val+c.val        - Type::Z);               }
    const CircVal  operator- (const CircVal& c) const { return Wrap(val-c.val        + Type::Z);               }
    const CircVal  operator* (const double&  r) const { return Wrap((val-Type::Z)*r  + Type::Z);               }
    const CircVal  operator/ (const double&  r) const { return Wrap((val-Type::Z)/r  + Type::Z);               }

          CircVal& operator+=(const CircVal& c)       { val= Wrap(val+c.val          - Type::Z); return *this; }
          CircVal& operator-=(const CircVal& c)       { val= Wrap(val-c.val          + Type::Z); return *this; }
          CircVal& operator*=(const double&  r)       { val= Wrap((val-Type::Z)*r    + Type::Z); return *this; }
          CircVal& operator/=(const double&  r)       { val= Wrap((val-Type::Z)/r    + Type::Z); return *this; }

          CircVal& operator =(const CircVal& c)       { val= c.val                             ; return *this; }

    bool           operator==(const CircVal& c) const { return val == c.val;                                   }
    bool           operator!=(const CircVal& c) const { return val != c.val;                                   }

    // note that two circular values can be compared in several different ways.
    // check carefully if this is really what you need!
    bool           operator> (const CircVal& c) const { return val >  c.val;                                   }
    bool           operator>=(const CircVal& c) const { return val >= c.val;                                   }
    bool           operator< (const CircVal& c) const { return val <  c.val;                                   }
    bool           operator<=(const CircVal& c) const { return val <= c.val;                                   }
};

// ==========================================================================
template <typename Type> static double        sin  (const CircVal<Type>& c) { return std::sin(ToR(CircVal<SignedRadRange>(c)));  }
template <typename Type> static double        cos  (const CircVal<Type>& c) { return std::cos(ToR(CircVal<SignedRadRange>(c)));  }
template <typename Type> static double        tan  (const CircVal<Type>& c) { return std::tan(ToR(CircVal<SignedRadRange>(c)));  }
template <typename Type> static CircVal<Type> asin (double r              ) { return CircVal<SignedRadRange>(std::asin (r    )); } // calls copy ctor CircVal(CircVal<SignedRadRange>)
template <typename Type> static CircVal<Type> acos (double r              ) { return CircVal<SignedRadRange>(std::acos (r    )); } // calls copy ctor CircVal(CircVal<SignedRadRange>)
template <typename Type> static CircVal<Type> atan (double r              ) { return CircVal<SignedRadRange>(std::atan (r    )); } // calls copy ctor CircVal(CircVal<SignedRadRange>)
template <typename Type> static CircVal<Type> atan2(double r1, double r2  ) { return CircVal<SignedRadRange>(std::atan2(r1,r2)); } // calls copy ctor CircVal(CircVal<SignedRadRange>)
template <typename Type> static CircVal<Type> ToC  (double r              ) { return CircVal<Type>::Wrap(r + Type::Z);           } // convert real-value r to circular-value in the range. 0 is converted to Type.Z

// ==========================================================================
// tester for CircVal class
template <typename Type>
class CircValTester
{
    // check if 2 circular-values are almost equal
    static bool IsCircAlmostEq(const CircVal<Type>& _f, const CircVal<Type>& _g)
    {
        double f= _f;
        double g= _g;

        if (::IsAlmostEq(f, g))
            return true;

        if (f < g)
            return IsAlmostEq(f, g - Type::R);
        else
            return IsAlmostEq(f, g + Type::R);
    }

    // assert that 2 circular-values are almost equal
    static void AssertCircAlmostEq(const CircVal<Type>& f, const CircVal<Type>& g)
    {
        assert(IsCircAlmostEq(f, g));
    }

    static void Test()
    {
        CircVal<Type> ZeroVal= Type::Z;

        // --------------------------------------------------------
        AssertCircAlmostEq(ZeroVal       , -ZeroVal);

        AssertAlmostEq    (sin(ZeroVal)  , 0.      );
        AssertAlmostEq    (cos(ZeroVal)  , 1.      );
        AssertAlmostEq    (tan(ZeroVal)  , 0.      );

        AssertCircAlmostEq(asin<Type>(0.), ZeroVal );
        AssertCircAlmostEq(acos<Type>(1.), ZeroVal );
        AssertCircAlmostEq(atan<Type>(0.), ZeroVal );

        AssertCircAlmostEq(ToC<Type>(0)  , ZeroVal );
        AssertAlmostEq    (ToR(ZeroVal)  , 0.      );

        // --------------------------------------------------------
        std::default_random_engine             rand_engine                 ;
        std::uniform_real_distribution<double> c_uni_dist(Type::L, Type::H);
        std::uniform_real_distribution<double> r_uni_dist(0.     , 1000.  ); // for multiplication,division by real-value
        std::uniform_real_distribution<double> t_uni_dist(-1.    , 1.     ); // for inverse-trigonometric functions

        std::random_device rnd_device;
        rand_engine.seed(rnd_device()); // reseed engine

        for (unsigned i= 10000; i--;)
        {
            CircVal<Type> c1(c_uni_dist(rand_engine)); // random circular value
            CircVal<Type> c2(c_uni_dist(rand_engine)); // random circular value
            CircVal<Type> c3(c_uni_dist(rand_engine)); // random circular value
            double        r (r_uni_dist(rand_engine)); // random real     value [    0, 1000) - for testing *,/ operators
            double        a1(t_uni_dist(rand_engine)); // random real     value [   -1,    1) - for testing asin,acos
            double        a2(t_uni_dist(rand_engine)); // random real     value [-1000, 1000) - for testing atan

            assert            (c1                                 == CircVal<Type>((double)c1)         );

            AssertCircAlmostEq(+c1                                  , c1                               ); // +c         = c
            AssertCircAlmostEq(-(-c1)                               , c1                               ); // -(-c)      = c
            AssertCircAlmostEq(c1 + c2                              , c2 + c1                          ); // c1+c2      = c2+c1
            AssertCircAlmostEq(c1 + (c2 +c3)                        , (c1 + c2) + c3                   ); // c1+(c2+c3) = (c1+c2)+c3
            AssertCircAlmostEq(c1 + -c1                             , ZeroVal                          ); // c+(-c)     = z
            AssertCircAlmostEq(c1 + ZeroVal                         , c1                               ); // c+z        = c

            AssertCircAlmostEq(c1      -  c1                        , ZeroVal                          ); // c-c        = z
            AssertCircAlmostEq(c1      - ZeroVal                    , c1                               ); // c-z        = c
            AssertCircAlmostEq(ZeroVal - c1                         , -c1                              ); // z-c        = -c
            AssertCircAlmostEq(c1      - c2                         , -(c2 - c1)                       ); // c1-c2      = -(c2-c1)

            AssertCircAlmostEq(c1 * 0.                              , ZeroVal                          ); // c*0        = 0
            AssertCircAlmostEq(c1 * 1.                              , c1                               ); // c*1        = c
            AssertCircAlmostEq(c1 / 1.                              , c1                               ); // c/1        = c

            AssertCircAlmostEq((c1 * (1./(r+1.))) / (1./(r+1.))     , c1                               ); // (c*r)/r    = c, 0<r<=1
            AssertCircAlmostEq((c1 / (    r+1.) ) * (    r+1. )     , c1                               ); // (c/r)*r    = c,   r>=1

            // --------------------------------------------------------
            AssertCircAlmostEq(~(~c1)                               , c1                               ); // opposite(opposite(c) = c
            AssertCircAlmostEq(c1 - (~c1)                           , ToC<Type>(Type::R/2.)            ); // c - ~c               = r/2+z

            // --------------------------------------------------------
            AssertAlmostEq    (sin(ToR(CircVal<SignedRadRange>(c1))),  sin(c1)                         ); // member func sin
            AssertAlmostEq    (cos(ToR(CircVal<SignedRadRange>(c1))),  cos(c1)                         ); // member func cos
            AssertAlmostEq    (tan(ToR(CircVal<SignedRadRange>(c1))),  tan(c1)                         ); // member func tan

            AssertAlmostEq    (sin(-c1)                             , -sin(c1)                         ); // sin(-c)    = -sin(c)
            AssertAlmostEq    (cos(-c1)                             ,  cos(c1)                         ); // cos(-c)    =  cos(c)
            AssertAlmostEq    (tan(-c1)                             , -tan(c1)                         ); // tan(-c1)   = -tan(c) the error may be large

            AssertAlmostEq    (sin(c1+ToC<Type>(Type::R/4.))        ,  cos(c1)                         ); // sin(c+r/4) =  cos(c)
            AssertAlmostEq    (cos(c1+ToC<Type>(Type::R/4.))        , -sin(c1)                         ); // cos(c+r/4) = -sin(c)
            AssertAlmostEq    (sin(c1+ToC<Type>(Type::R/2.))        , -sin(c1)                         ); // sin(c+r/2) = -sin(c)
            AssertAlmostEq    (cos(c1+ToC<Type>(Type::R/2.))        , -cos(c1)                         ); // cos(c+r/2) = -cos(c)

            AssertAlmostEq    (Sqr(sin(c1))+Sqr(cos(c1))            , 1.                               ); // sin(x)^2+cos(x)^2 = 1

            AssertAlmostEq    (sin(c1)/cos(c1)                      , tan(c1)                          ); // sin(x)/cos(x) = tan(x)

            // --------------------------------------------------------
            AssertCircAlmostEq(asin<Type>(a1)                       , CircVal<SignedRadRange>(asin(a1))); // member func asin
            AssertCircAlmostEq(acos<Type>(a1)                       , CircVal<SignedRadRange>(acos(a1))); // member func acos
            AssertCircAlmostEq(atan<Type>(a2)                       , CircVal<SignedRadRange>(atan(a2))); // member func atan

            AssertCircAlmostEq(asin<Type>(a1) + asin<Type>(-a1)     , ZeroVal                          ); // asin(r)+asin(-r) = z
            AssertCircAlmostEq(acos<Type>(a1) + acos<Type>(-a1)     , ToC<Type>(Type::R/2.)            ); // acos(r)+acos(-r) = r/2+z
            AssertCircAlmostEq(asin<Type>(a1) + acos<Type>( a1)     , ToC<Type>(Type::R/4.)            ); // asin(r)+acos( r) = r/4+z
            AssertCircAlmostEq(atan<Type>(a2) + atan<Type>(-a2)     , ZeroVal                          ); // atan(r)+atan(-r) = z

            // --------------------------------------------------------
            assert            (c1 >  c2                           ==    (c2 <  c1)                     ); // c1> c2 <==>   c2< c1
            assert            (c1 >= c2                           ==    (c2 <= c1)                     ); // c1>=c2 <==>   c2<=c1
            assert            (c1 >= c2                           ==  ( (c1 >  c2) ||  (c1 == c2))     ); // c1>=c2 <==>  (c1> c2)|| (c1==c2)
            assert            (c1 <= c2                           ==  ( (c1 <  c2) ||  (c1 == c2))     ); // c1<=c2 <==>  (c1< c2)|| (c1==c2)
            assert            (c1 >  c2                           ==  (!(c1 == c2) && !(c1 <  c2))     ); // c1> c2 <==> !(c1==c2)&&!(c1< c2)
            assert            (c1 == c2                           ==  (!(c1 >  c2) && !(c1 <  c2))     ); // c1= c2 <==> !(c1> c2)&&!(c1< c2)
            assert            (c1 <  c2                           ==  (!(c1 == c2) && !(c1 >  c2))     ); // c1< c2 <==> !(c1==c2)&&!(c1> c2)
            assert            (!(c1>c2) || !(c2>c3) || (c1>c3)                                         ); // (c1>c2)&&(c2>c3) ==> c1>c3

            // --------------------------------------------------------
            AssertCircAlmostEq(c1                                   , ToC<Type>(ToR( c1)       )       ); //  c1        = ToC(ToR( c1)
            AssertCircAlmostEq(-c1                                  , ToC<Type>(ToR(-c1)       )       ); // -c1        = ToC(ToR(-c1)
            AssertCircAlmostEq(c1 + c2                              , ToC<Type>(ToR(c1)+ToR(c2))       ); // c1+c2      = ToC(ToR(c1)+ToR(c2))
            AssertCircAlmostEq(c1 - c2                              , ToC<Type>(ToR(c1)-ToR(c2))       ); // c1-c2      = ToC(ToR(c1)-ToR(c2))
            AssertCircAlmostEq(c1 * r                               , ToC<Type>(ToR(c1)*r      )       ); // c1*r       = ToC(ToR(c1)*r      )
            AssertCircAlmostEq(c1 / r                               , ToC<Type>(ToR(c1)/r      )       ); // c1/r       = ToC(ToR(c1)/r      )

            // --------------------------------------------------------
        }
    }

public:
    CircValTester()
    {
        Test();
    }
};

