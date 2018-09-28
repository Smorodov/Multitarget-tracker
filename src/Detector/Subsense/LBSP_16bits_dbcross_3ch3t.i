// note: this is the LBSP 16 bit double-cross indiv RGB pattern as used in
// the original article by G.-A. Bilodeau et al.
//
//  O   O   O          4 ..  3 ..  6
//    O O O           .. 15  8 13 ..
//  O O X O O    =>    0  9  X 11  1
//    O O O           .. 12 10 14 ..
//  O   O   O          7 ..  2 ..  5
//           3x                     3x
//
// must be defined externally:
//      _t              (size_t[3], absolute thresholds used for comparisons)
//      _ref            (uchar[3], 'central' values used for comparisons)
//      _data           (uchar*, triple-channel data to be covered by the pattern)
//      _y              (int, pattern rows location in the image data)
//      _x              (int, pattern cols location in the image data)
//      _step_row       (size_t, step size between rows, including padding)
//      _res            (ushort[3], 16 bit result vectors vector)
//       L1dist         (function, returns the absolute difference between two uchars)

#ifdef _val
#error "definitions clash detected"
#else
#define _val(x,y,n) _data[_step_row*(_y+y)+3*(_x+x)+n]
#endif

for(int n=0; n<3; ++n) {
    _res[n] = ((L1dist(_val(-1, 1, n),_ref[n]) > _t[n]) << 15)
            + ((L1dist(_val( 1,-1, n),_ref[n]) > _t[n]) << 14)
            + ((L1dist(_val( 1, 1, n),_ref[n]) > _t[n]) << 13)
            + ((L1dist(_val(-1,-1, n),_ref[n]) > _t[n]) << 12)
            + ((L1dist(_val( 1, 0, n),_ref[n]) > _t[n]) << 11)
            + ((L1dist(_val( 0,-1, n),_ref[n]) > _t[n]) << 10)
            + ((L1dist(_val(-1, 0, n),_ref[n]) > _t[n]) << 9)
            + ((L1dist(_val( 0, 1, n),_ref[n]) > _t[n]) << 8)
            + ((L1dist(_val(-2,-2, n),_ref[n]) > _t[n]) << 7)
            + ((L1dist(_val( 2, 2, n),_ref[n]) > _t[n]) << 6)
            + ((L1dist(_val( 2,-2, n),_ref[n]) > _t[n]) << 5)
            + ((L1dist(_val(-2, 2, n),_ref[n]) > _t[n]) << 4)
            + ((L1dist(_val( 0, 2, n),_ref[n]) > _t[n]) << 3)
            + ((L1dist(_val( 0,-2, n),_ref[n]) > _t[n]) << 2)
            + ((L1dist(_val( 2, 0, n),_ref[n]) > _t[n]) << 1)
            + ((L1dist(_val(-2, 0, n),_ref[n]) > _t[n]));
}

#undef _val
