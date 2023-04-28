// ==========================================================================
// truncated normal distribution
// Lior Kogan (koganlior1@gmail.com), 2012
// based on VC 2012 std::normal_distribution (random) as a skeleton
// and on C. H. Jackson's R's implementation of the following paper:
// Robert, C. P. Simulation of truncated normal variables. Statistics and Computing (1995) 5, 121-125
// ==========================================================================

#pragma once

// ==========================================================================
// TEMPLATE CLASS truncated_normal_distribution
template<class _Ty= double>
class truncated_normal_distribution
{   // template class for truncated normal distribution
public:
    typedef truncated_normal_distribution<_Ty> _Myt;
    typedef _Ty input_type ;
    typedef _Ty result_type;

    struct param_type
    {   // parameter package
        typedef _Myt distribution_type;

        param_type(_Ty _Mean0= 0., _Ty _Sigma0= 1., _Ty _A0= 0., _Ty _B0= 0.)
        {   // construct from parameters
            _Init(_Mean0, _Sigma0, _A0, _B0);
        }

        bool operator==(const param_type& _Right) const
        {   // test for equality
            return _Mean  == _Right._Mean  &&
                   _Sigma == _Right._Sigma &&
                   _A     == _Right._A     &&
                   _B     == _Right._B        ;
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

        _Ty a() const
        {   // return truncation-range lower-bound
            return _A;
        }

        _Ty b() const
        {   // return truncation-range upper-bound
            return _B;
        }

        _Ty stddev() const
        {   // return sigma value
            return _Sigma;
        }

        int alg() const
        {  // return fastest algorithm for the given parameters
            return _Alg;
        }

        void _Init(_Ty _Mean0, _Ty _Sigma0, _Ty _A0, _Ty _B0)
        {   // set internal state
            _RNG_ASSERT(0.  < _Sigma0, "invalid sigma argument for truncated_normal_distribution");
            _RNG_ASSERT(_A0 < _B0    , "invalid truncation-range for truncated_normal_distribution");
            _Mean = _Mean0 ;
            _Sigma= _Sigma0;
            _A    = _A0    ;
            _B    = _B0    ;

            _NA= (_A - _Mean) / _Sigma;
            _NB= (_B - _Mean) / _Sigma;

            // decide on the fastest algorithm for our case
            _Alg= 3;
                 if ((_NA < 0 ) && ( _NB > 0) && (_NB - _NA > sqrt(_2Pi)))                                                                        _Alg= 0;
            else if ((_NA >= 0) && ( _NB >  _NA + 2.*sqrt(exp(1.)) / ( _NA + sqrt(Sqr(_NA) + 4.)) * exp((_NA*2. -  _NA*sqrt(Sqr(_NA) + 4.))/4.))) _Alg= 1;
            else if ((_NB <= 0) && (-_NA > -_NB + 2.*sqrt(exp(1.)) / (-_NB + sqrt(Sqr(_NB) + 4.)) * exp((_NB*2. - -_NB*sqrt(Sqr(_NB) + 4.))/4.))) _Alg= 2;
        }

        _Ty _Mean ;
        _Ty _Sigma;
        _Ty _A    ;
        _Ty _B    ;

        _Ty _NA   ; // _A normalized
        _Ty _NB   ; // _B normalized
        int _Alg  ; // algorithm to use
    };

    explicit truncated_normal_distribution(_Ty _Mean0 = 0.                              ,
                                           _Ty _Sigma0= 1.                              ,
                                           _Ty _A0    = std::numeric_limits< _Ty>::min(),  // truncation-range lower-bound
                                           _Ty _B0    = std::numeric_limits< _Ty>::max() ) // truncation-range upper-bound

        : _Par(_Mean0, _Sigma0, _A0, _B0), _Valid(false), _X2(0)
    {   // construct
    }

    explicit truncated_normal_distribution(param_type _Par0)
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

    _Ty a() const
    {   // return truncation-range lower-bound
        return _Par.a();
    }

    _Ty b() const
    {   // return truncation-range upper-bound
        return _Par.b();
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
        return _Par._A;
    }

    result_type (max)() const
    {   // get largest possible result
        return _Par._B;
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
        _Ty _A0    ;
        _Ty _B0    ;
        _In(_Istr, _Mean0 );
        _In(_Istr, _Sigma0);
        _In(_Istr, _A0    );
        _In(_Istr, _B0    );
        _Par._Init(_Mean0, _Sigma0, _A0, _B0);

        _Istr >> _Valid;
        _In(_Istr, _X2);
        return _Istr;
    }

    template<class _Elem, class _Traits>
    basic_ostream<_Elem, _Traits>& _Write(basic_ostream<_Elem, _Traits>& _Ostr) const
    {   // write state to _Ostr
        _Out(_Ostr, _Par._Mean );
        _Out(_Ostr, _Par._Sigma);
        _Out(_Ostr, _Par._A    );
        _Out(_Ostr, _Par._B    );

        _Ostr << ' ' << _Valid;
        _Out(_Ostr, _X2);
        return _Ostr;
    }

private:
    template<class _Engine>
    result_type _Eval(_Engine& _Eng, const param_type& _Par0, bool _Keep= true)
    {
        _Ty r;

        switch (_Par0._Alg)
        {
        case 0 :
            {
                normal_distribution<_Ty> nd;
                do  { r= nd(_Eng); }
                while (r<_Par0._NA || r>_Par0._NB);
                break;
            }

        case 1 :
            {
                exponential_distribution<_Ty> ed;
                _Ty a,u,z;

                do
                {
                    a= (_Par0._NA + sqrt(Sqr(_Par0._NA)+4.))/2.;
                    z= ed(_Eng, a) + _Par0._NA;
                    u= _NRAND(_Eng, _Ty);
                }
                while ((u>exp(-Sqr(z-a)/2.)) || (z>_Par0._NB));

                r= z;
                break;
            }

        case 2 :
            {
                exponential_distribution<_Ty> ed;
                _Ty a,u,z;

                do
                {
                    a= (-_Par0._NB + sqrt(Sqr(_Par0._NB)+4.))/2.;
                    z= ed(_Eng, a) - _Par0._NB;
                    u= _NRAND(_Eng, _Ty);
                }
                while ((u>exp(-Sqr(z-a)/2.)) || (z>-_Par0._NA));

                r= -z;
                break;
            }

        default:
            {
                _Ty z,u,rho;

                do
                {
                    uniform_real<_Ty> ud(_Par0._NA, _Par0._NB);
                    z= ud(_Eng);
                    u= _NRAND(_Eng, _Ty);

                         if (_Par0._NA>0) rho= exp((Sqr(_Par0._NA)-Sqr(z))/2.);
                    else if (_Par0._NB<0) rho= exp((Sqr(_Par0._NB)-Sqr(z))/2.);
                    else                  rho= exp(               -Sqr(z) /2.);
                }
                while (u>rho);

                r= z;
            }
        }

        return r * _Par0._Sigma + _Par0._Mean; // denormalize result
    }

    int        _Alg  ; // which algorithm to use
    param_type _Par  ;
    bool       _Valid;
    _Ty        _X2   ;
};

template<class _Elem, class _Traits, class _Ty>
basic_istream<_Elem, _Traits>& operator>>(basic_istream<_Elem, _Traits>& _Istr, truncated_normal_distribution<_Ty>& _Dist)
{   // read state from _Istr
    return _Dist._Read(_Istr);
}

template<class _Elem, class _Traits, class _Ty>
basic_ostream<_Elem, _Traits>& operator<<(basic_ostream<_Elem, _Traits>& _Ostr, const truncated_normal_distribution<_Ty>& _Dist)
{   // write state to _Ostr
    return _Dist._Write(_Ostr);
}
