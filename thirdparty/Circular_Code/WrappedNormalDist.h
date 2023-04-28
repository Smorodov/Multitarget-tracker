// ==========================================================================
// wrapped normal distribution
// Lior Kogan (koganlior1@gmail.com), 2012
// based on VC 2012 std::normal_distribution (random) as a skeleton
// ==========================================================================

#pragma once

#include "CircHelper.h" // Mod

// ==========================================================================
// TEMPLATE CLASS wrapped_normal_distribution
template<class _Ty= double>
class wrapped_normal_distribution
{   // template class for wrapped normal distribution
public:
    typedef wrapped_normal_distribution<_Ty> _Myt;
    typedef _Ty input_type ;
    typedef _Ty result_type;

    struct param_type
    {   // parameter package
        typedef _Myt distribution_type;

        param_type(_Ty _Mean0= 0., _Ty _Sigma0= 1., _Ty _L0= 0., _Ty _H0= 0.)
        {   // construct from parameters
            _Init(_Mean0, _Sigma0, _L0, _H0);
        }

        bool operator==(const param_type& _Right) const
        {   // test for equality
            return _Mean  == _Right._Mean  &&
                   _Sigma == _Right._Sigma &&
                   _L     == _Right._L     &&
                   _H     == _Right._H        ;

        }

        bool operator!=(const param_type& _Right) const
        {   // test for inequality
            return !(*this == _Right);
        }

        _Ty mean() const
        {   // return mean value
            return _Mean;
        }

        _Ty sigma() const
        {   // return sigma value
            return _Sigma;
        }

        _Ty l() const
        {   // return wrapping-range lower-bound
            return _L;
        }

        _Ty h() const
        {   // return wrapping-range upper-bound
            return _H;
        }

        _Ty stddev() const
        {   // return sigma value
            return _Sigma;
        }

        void _Init(_Ty _Mean0, _Ty _Sigma0, _Ty _L0, _Ty _H0)
        {   // set internal state
            _RNG_ASSERT(0.  < _Sigma0, "invalid sigma argument for wrapped_normal_distribution");
            _RNG_ASSERT(_L0 < _H0    , "invalid wrapping-range for wrapped_normal_distribution");
            _Mean = _Mean0 ;
            _Sigma= _Sigma0;
            _L    = _L0    ;
            _H    = _H0    ;
        }

        _Ty _Mean ;
        _Ty _Sigma;
        _Ty _L    ;
        _Ty _H    ;
    };

    explicit wrapped_normal_distribution(_Ty _Mean0 =    0.,
                                         _Ty _Sigma0=   45.,
                                         _Ty _L0    = -180.,  // wrapping-range lower-bound
                                         _Ty _H0    =  180. ) // wrapping-range upper-bound
        : _Par(_Mean0, _Sigma0, _L0, _H0), _Valid(false), _X2(0)
    {   // construct
    }

    explicit wrapped_normal_distribution(param_type _Par0)
        : _Par(_Par0), _Valid(false), _X2(0)
    {   // construct from parameter package
    }

    _Ty mean() const
    {   // return mean value
        return _Par.mean();
    }

    _Ty sigma() const
    {   // return sigma value
        return _Par.sigma();
    }

    _Ty l() const
    {   // return wrapping-range lower-bound
        return _Par.l();
    }

    _Ty h() const
    {   // return wrapping-range upper-bound
        return _Par.h();
    }

    _Ty stddev() const
    {   // return sigma value
        return _Par.sigma();
    }

    param_type param() const
    {   // return parameter package
        return _Par;
    }

    void param(const param_type& _Par0)
    {   // set parameter package
        _Par= _Par0;
        reset();
    }

    result_type (min)() const
    {   // get smallest possible result
        return _Par._L;
    }

    result_type (max)() const
    {   // get largest possible result
        return _Par._H;
    }

    void reset()
    {   // clear internal state
        _Valid= false;
    }

    template<class _Engine>
    result_type operator()(_Engine& _Eng)
    {   // return next value
        return _Eval(_Eng, _Par);
    }

    template<class _Engine>
    result_type operator()(_Engine& _Eng, const param_type& _Par0)
    {   // return next value, given parameter package
        reset();
        return _Eval(_Eng, _Par0, false);
    }

    template<class _Elem, class _Traits>
    basic_istream<_Elem, _Traits>& _Read(basic_istream<_Elem, _Traits>& _Istr)
    {   // read state from _Istr
        _Ty _Mean0 ;
        _Ty _Sigma0;
        _Ty _L0    ;
        _Ty _H0    ;
        _In(_Istr, _Mean0 );
        _In(_Istr, _Sigma0);
        _In(_Istr, _L0    );
        _In(_Istr, _H0    );
        _Par._Init(_Mean0, _Sigma0, _L0, _H0);

        _Istr >> _Valid;
        _In(_Istr, _X2);
        return _Istr;
    }

    template<class _Elem, class _Traits>
    basic_ostream<_Elem, _Traits>& _Write(basic_ostream<_Elem, _Traits>& _Ostr) const
    {   // write state to _Ostr
        _Out(_Ostr, _Par._Mean );
        _Out(_Ostr, _Par._Sigma);
        _Out(_Ostr, _Par._L    );
        _Out(_Ostr, _Par._H    );

        _Ostr << ' ' << _Valid;
        _Out(_Ostr, _X2);
        return _Ostr;
    }

private:
    template<class _Engine> result_type _Eval(_Engine& _Eng, const param_type& _Par0, bool _Keep= true)
    {   // compute next value
        // Knuth, vol. 2, p. 122, alg. P
        _Ty r;

        if (_Keep && _Valid)
        {   // return stored value
            r     = _X2  ;
            _Valid= false;
        }
        else
        {   // generate two values, store one, return one
            double _V1, _V2, _Sx;
            for (; ; )
            {   // reject bad values
                _V1= 2 * _NRAND(_Eng, _Ty) - 1.;
                _V2= 2 * _NRAND(_Eng, _Ty) - 1.;
                _Sx= _V1 * _V1 + _V2 * _V2;
                if (_Sx < 1.)
                    break;
            }

            double _Fx= _CSTD sqrt(-2. * _CSTD log(_Sx) / _Sx);
            if (_Keep)
            {   // save second value for next call
                _X2   = _Fx * _V2;
                _Valid= true     ;
            }

            r= _Fx * _V1;
        }

        result_type d= r * _Par0._Sigma + _Par0._Mean;            // denormalize result
        return Mod(d - _Par0._L, _Par0._H - _Par0._L) + _Par0._L; // wrap        result
    }

    param_type _Par  ;
    bool       _Valid;
    _Ty        _X2   ;
};

template<class _Elem, class _Traits, class _Ty>
basic_istream<_Elem, _Traits>& operator>>(basic_istream<_Elem, _Traits>& _Istr, wrapped_normal_distribution<_Ty>& _Dist)
{   // read state from _Istr
    return _Dist._Read(_Istr);
}

template<class _Elem, class _Traits, class _Ty>
basic_ostream<_Elem, _Traits>& operator<<(basic_ostream<_Elem, _Traits>& _Ostr, const wrapped_normal_distribution<_Ty>& _Dist)
{   // write state to _Ostr
    return _Dist._Write(_Ostr);
}
