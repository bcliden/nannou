use float_eq::assert_float_eq;
use nannou_core::math::fmod;
use num_traits::Float;

#[test]
fn test_fmod() {
    assert_float_eq!(fmod(5.1, 3.0), 2.1, r2nd <= Float::epsilon());
    assert_float_eq!(fmod(-5.1, 3.0), -2.1, r2nd <= Float::epsilon());
    assert_float_eq!(fmod(5.1, -3.0), 2.1, r2nd <= Float::epsilon());
    assert_float_eq!(fmod(-5.1, -3.0), -2.1, r2nd <= Float::epsilon());

    assert_float_eq!(fmod(0.0, 1.0), 0.0, r2nd <= Float::epsilon());
    assert_float_eq!(fmod(-0.0, 1.0), -0.0, r2nd <= Float::epsilon());

    assert!(fmod(5.1, Float::infinity()).is_sign_negative());
    assert!(fmod(5.1, Float::infinity()).is_nan());
}
