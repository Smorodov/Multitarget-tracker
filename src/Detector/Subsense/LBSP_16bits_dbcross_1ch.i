// note: this is the LBSP 16 bit double-cross single channel pattern as used in
// the original article by G.-A. Bilodeau et al.
// 
//  O   O   O          4 ..  3 ..  6
//    O O O           .. 15  8 13 ..
//  O O X O O    =>    0  9  X 11  1
//    O O O           .. 12 10 14 ..
//  O   O   O          7 ..  2 ..  5
//
//
// must be defined externally:
//      _t              (size_t, absolute threshold used for comparisons)
//      _ref            (uchar, 'central' value used for comparisons)
//      _data           (uchar*, single-channel data to be covered by the pattern)
//      _y              (int, pattern rows location in the image data)
//      _x              (int, pattern cols location in the image data)
//      _step_row       (size_t, step size between rows, including padding)
//      _res            (ushort, 16 bit result vector)
//       L1dist         (function, returns the absolute difference between two uchars)

#ifdef _val
#error "definitions clash detected"
#else
#define _val(x,y) _data[_step_row*(_y+y)+_x+x]
#endif

_res = ((L1dist(_val(-1, 1),_ref) > _t) << 15)
     + ((L1dist(_val( 1,-1),_ref) > _t) << 14)
     + ((L1dist(_val( 1, 1),_ref) > _t) << 13)
     + ((L1dist(_val(-1,-1),_ref) > _t) << 12)
     + ((L1dist(_val( 1, 0),_ref) > _t) << 11)
     + ((L1dist(_val( 0,-1),_ref) > _t) << 10)
     + ((L1dist(_val(-1, 0),_ref) > _t) << 9)
     + ((L1dist(_val( 0, 1),_ref) > _t) << 8)
     + ((L1dist(_val(-2,-2),_ref) > _t) << 7)
     + ((L1dist(_val( 2, 2),_ref) > _t) << 6)
     + ((L1dist(_val( 2,-2),_ref) > _t) << 5)
     + ((L1dist(_val(-2, 2),_ref) > _t) << 4)
     + ((L1dist(_val( 0, 2),_ref) > _t) << 3)
     + ((L1dist(_val( 0,-2),_ref) > _t) << 2)
     + ((L1dist(_val( 2, 0),_ref) > _t) << 1)
     + ((L1dist(_val(-2, 0),_ref) > _t));

#undef _val
