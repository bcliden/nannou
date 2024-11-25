//! A mathematical foundation for nannou including point and vector types and a range of
//! helper/utility functions.

use core::ops::Add;
use num_traits::{Float, NumCast, One};

pub use num_traits;

const ONE_TURN_DEGREES_F32: f32 = 360.0;
const ONE_TURN_DEGREES_F64: f64 = 360.0;

/// Functions for converting between angle representations.
///
/// Only implemented for `f32` and `f64`.
pub trait ConvertAngle {
    fn deg_to_rad(self) -> Self;
    fn rad_to_deg(self) -> Self;
    fn turns_to_rad(self) -> Self;
    fn rad_to_turns(self) -> Self;
}

/// Short-hand for retrieving the angle of the vector in radians.
pub trait Vec2Angle {
    /// The angle of the vector in radians.
    ///
    /// Implemented internally as `glam::Vec2::X.angle_between(self)`.
    fn angle(self) -> f32;
}

pub trait Vec2Rotate {
    /// Rotate the vector around 0.0.
    fn rotate(self, radians: f32) -> Self;
}

/// Create a transformation matrix that will cause a vector to point at `dir` using `up` for
/// orientation.
// NOTE: Remove this if we can land something similar upstream in `glam`.
pub trait Mat4LookTo {
    fn look_to_rh(eye: glam::Vec3, dir: glam::Vec3, up: glam::Vec3) -> glam::Mat4 {
        let f = dir.normalize();
        let s = f.cross(up).normalize();
        let u = s.cross(f);
        glam::Mat4::from_cols(
            glam::vec4(s.x, u.x, -f.x, 0.0),
            glam::vec4(s.y, u.y, -f.y, 0.0),
            glam::vec4(s.z, u.z, -f.z, 0.0),
            glam::vec4(-eye.dot(s), -eye.dot(u), eye.dot(f), 1.0),
        )
    }

    fn look_to_lh(eye: glam::Vec3, dir: glam::Vec3, up: glam::Vec3) -> glam::Mat4 {
        Self::look_to_rh(eye, -dir, up)
    }
}

impl ConvertAngle for f32 {
    /// # Examples
    /// ```
    /// # use nannou_core::prelude::*;
    /// # use nannou_core::math::ConvertAngle;
    /// assert_eq!(90_f32.deg_to_rad(), core::f32::consts::PI / 2.0);
    /// assert_eq!(-90_f32.deg_to_rad(), -core::f32::consts::PI / 2.0);
    /// assert_eq!(180_f32.deg_to_rad(), core::f32::consts::PI);
    /// assert_eq!(-180_f32.deg_to_rad(), -core::f32::consts::PI);
    /// assert_eq!(360_f32.deg_to_rad(), 2.0 * core::f32::consts::PI);
    /// ```
    fn deg_to_rad(self) -> Self {
        self * core::f32::consts::TAU / ONE_TURN_DEGREES_F32
    }

    /// # Examples
    /// ```
    /// # use nannou_core::prelude::*;
    /// # use nannou_core::math::ConvertAngle;
    /// assert_eq!((core::f32::consts::PI / 2.0).rad_to_deg(), 90_f32);
    /// assert_eq!((-(core::f32::consts::PI / 2.0)).rad_to_deg(), -90_f32);
    /// assert_eq!(core::f32::consts::PI.rad_to_deg(), 180_f32);
    /// assert_eq!(-(core::f32::consts::PI).rad_to_deg(), -180_f32);
    /// assert_eq!((core::f32::consts::TAU).rad_to_deg(), 360_f32);
    /// ```
    fn rad_to_deg(self) -> Self {
        self * ONE_TURN_DEGREES_F32 / core::f32::consts::TAU
    }

    /// # Examples
    /// ```
    /// # use nannou_core::prelude::*;
    /// # use nannou_core::math::ConvertAngle;
    ///
    /// assert_eq!(0.5_f32.turns_to_rad(), core::f32::consts::PI);
    /// assert_eq!(1.0_f32.turns_to_rad(), core::f32::consts::TAU);
    /// assert_eq!(2.0_f32.turns_to_rad(), 2.0 * core::f32::consts::TAU);
    /// assert_eq!(-2.0_f32.turns_to_rad(), -2.0 * core::f32::consts::TAU);
    /// ```
    fn turns_to_rad(self) -> Self {
        self * core::f32::consts::TAU
    }

    /// # Examples
    /// ```
    /// # use nannou_core::prelude::*;
    /// # use nannou_core::math::ConvertAngle;
    /// assert_eq!(core::f32::consts::PI.rad_to_turns(), 0.5_f32);
    /// assert_eq!(core::f32::consts::TAU.rad_to_turns(), 1_f32);
    /// assert_eq!((2.0 * core::f32::consts::TAU).rad_to_turns(), 2_f32);
    /// assert_eq!((-2.0 * core::f32::consts::TAU).rad_to_turns(), -2_f32);
    /// ```
    fn rad_to_turns(self) -> Self {
        self / core::f32::consts::TAU
    }
}

impl ConvertAngle for f64 {
    fn deg_to_rad(self) -> Self {
        self * core::f64::consts::TAU / ONE_TURN_DEGREES_F64
    }
    fn rad_to_deg(self) -> Self {
        self * ONE_TURN_DEGREES_F64 / core::f64::consts::TAU
    }
    fn turns_to_rad(self) -> Self {
        self * core::f64::consts::TAU
    }
    fn rad_to_turns(self) -> Self {
        self / core::f64::consts::TAU
    }
}

impl Vec2Angle for glam::Vec2 {
    /// # Examples
    /// ```
    /// # use nannou_core::prelude::*;
    /// # use glam::Vec2;
    /// assert_eq!(Vec2::new(15.0, 5.0).angle(), 0.3217505_f32);
    /// assert_eq!(Vec2::new(200.0, 400.0).angle(), 1.1071486_f32);
    /// assert_eq!(Vec2::new(-10.0, 40.0).angle(), 1.815775_f32);
    /// ```
    fn angle(self) -> f32 {
        glam::Vec2::X.angle_between(self)
    }
}

impl Vec2Rotate for glam::Vec2 {
    fn rotate(self, radians: f32) -> Self {
        let rad_cos = radians.cos();
        let rad_sin = radians.sin();
        let x = self.x * rad_cos - self.y * rad_sin;
        let y = self.x * rad_sin + self.y * rad_cos;
        glam::vec2(x, y)
    }
}

impl Mat4LookTo for glam::Mat4 {}

/// Maps a value from an input range to an output range.
///
/// Note that `map_range` doesn't clamp the output: if `val` is outside the input range, the mapped
/// value will be outside the output range. (Use `clamp` to restrict the output, if desired.)
///
/// # Examples
/// ```
/// # use nannou_core::prelude::*;
/// assert_eq!(map_range(128, 0, 255, 0.0, 1.0), 0.5019607843137255);
/// ```
/// ```
/// # use nannou_core::prelude::*;
/// assert_eq!(map_range(3, 0, 10, 0.0, 1.0), 0.3);
/// ```
/// ```
/// # use nannou_core::prelude::*;
/// // When the value is outside the input range, the result will be outside the output range.
/// let result = map_range(15, 0, 10, 0.0, 1.0);
/// assert_eq!(result, 1.5);
/// assert_eq!(clamp(result, 0.0, 1.0), 1.0);
/// ```
// TODO: Should consider refactoring this to only allow for conversions between types that cannot
// fail. This would break some code but make users think about issues like converting to signed
// ranges with unsigned types, etc. Would also reduce size of code generated due to all the panic
// branches.
pub fn map_range<X, Y>(val: X, in_min: X, in_max: X, out_min: Y, out_max: Y) -> Y
where
    X: NumCast,
    Y: NumCast,
{
    macro_rules! unwrap_or_panic {
        ($result:expr, $arg:expr) => {
            $result.unwrap_or_else(|| panic!("[map_range] failed to cast {} arg to `f64`", $arg))
        };
    }

    let val_f: f64 = unwrap_or_panic!(NumCast::from(val), "first");
    let in_min_f: f64 = unwrap_or_panic!(NumCast::from(in_min), "second");
    let in_max_f: f64 = unwrap_or_panic!(NumCast::from(in_max), "third");
    let out_min_f: f64 = unwrap_or_panic!(NumCast::from(out_min), "fourth");
    let out_max_f: f64 = unwrap_or_panic!(NumCast::from(out_max), "fifth");

    NumCast::from((val_f - in_min_f) / (in_max_f - in_min_f) * (out_max_f - out_min_f) + out_min_f)
        .unwrap_or_else(|| panic!("[map_range] failed to cast result to target type"))
}

/// The max between two partially ordered values.
/// # Examples
/// ```
/// # use nannou_core::prelude::*;
/// assert_eq!(partial_max(5, 10), 10);
/// assert_eq!(partial_max(5, 5), 5);
/// assert_eq!(partial_max(5, 0), 5);
/// ```
pub fn partial_max<T>(a: T, b: T) -> T
where
    T: PartialOrd,
{
    if a >= b {
        a
    } else {
        b
    }
}

/// The min between two partially ordered values.
/// # Examples
/// ```
/// # use nannou_core::prelude::*;
/// assert_eq!(partial_min(5, 10), 5);
/// assert_eq!(partial_min(5, 5), 5);
/// assert_eq!(partial_min(5, 0), 0);
/// ```
pub fn partial_min<T>(a: T, b: T) -> T
where
    T: PartialOrd,
{
    if a <= b {
        a
    } else {
        b
    }
}

/// Clamp a value between some range.
/// # Examples
/// ```
/// # use nannou_core::prelude::*;
/// assert_eq!(clamp(-5, 0, 10), 0);
/// assert_eq!(clamp(15, 0, 10), 10);
/// assert_eq!(clamp(5, 0, 10), 5);
///
/// // Range start and end don't have to strictly be ordered
/// assert_eq!(clamp(-5, 10, 0), 0);
/// assert_eq!(clamp(15, 10, 0), 10);
/// assert_eq!(clamp(5, 10, 0), 5);
/// ```
pub fn clamp<T>(n: T, start: T, end: T) -> T
where
    T: PartialOrd,
{
    if start <= end {
        if n < start {
            start
        } else if n > end {
            end
        } else {
            n
        }
    } else {
        if n < end {
            end
        } else if n > start {
            start
        } else {
            n
        }
    }
}

/// # Examples
/// ```
/// # use nannou_core::math::two;
/// assert_eq!(two::<u32>(), 2u32);
/// assert_eq!(two::<u64>(), 2u64);
///
/// assert_eq!(two::<i32>(), 2i32);
/// assert_eq!(two::<i64>(), 2i64);
///
/// assert_eq!(two::<f32>(), 2f32);
/// assert_eq!(two::<f64>(), 2f64);
///
/// // Alternatively:
/// let float_test: f64 = two();
/// assert_eq!(float_test, 2f64);
/// ```
pub fn two<S>() -> S
where
    S: Add<Output = S> + One,
{
    S::one() + S::one()
}

/// Models the C++ fmod function.
/// # Examples
/// ```
/// # use nannou_core::prelude::*;
/// fn is_close_to<F>(f1: F, f2: F, delta: F) -> bool
///     where F: Float
/// {
///     f1 - f2 < delta && f2 - f1 < delta
/// }
/// let tolerance = 0.01;
///
/// assert!(is_close_to(fmod(5.1, 3.0), 2.1, tolerance));
/// assert!(is_close_to(fmod(-5.1, 3.0), 2.1, tolerance)); // should be -2.1
/// assert!(is_close_to(fmod(5.1, -3.0), -2.1, tolerance)); // should be 2.1
/// assert!(is_close_to(fmod(-5.1, -3.0), -2.1, tolerance));
///
/// assert_eq!(fmod(0.0, 1.0), 0.0);
/// assert_eq!(fmod(-0.0, 1.0), -0.0);
/// assert!(fmod(5.1, f32::INFINITY).is_sign_negative());
/// assert!(fmod(5.1, f32::INFINITY).is_nan());
/// ```
#[inline]
pub fn fmod<F>(numer: F, denom: F) -> F
where
    F: Float,
{
    let rquot: F = (numer / denom).floor();
    numer - rquot * denom
}

/// Convert the given angle in degrees to the same angle in radians.
pub fn deg_to_rad<S>(s: S) -> S
where
    S: ConvertAngle,
{
    s.deg_to_rad()
}

/// Convert the given angle in radians to the same angle in degrees.
pub fn rad_to_deg<S>(s: S) -> S
where
    S: ConvertAngle,
{
    s.rad_to_deg()
}

/// Convert the given value as a number of "turns" into the equivalent angle in radians.
pub fn turns_to_rad<S>(s: S) -> S
where
    S: ConvertAngle,
{
    s.turns_to_rad()
}

/// Convert the given value in radians to the equivalent value as a number of turns.
pub fn rad_to_turns<S>(s: S) -> S
where
    S: ConvertAngle,
{
    s.rad_to_turns()
}
