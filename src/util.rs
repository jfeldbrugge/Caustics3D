use hdf5;
// use self as util;

#[inline]
pub fn get_or_create_group<S>(parent: &hdf5::Group, name: S) -> hdf5::Result<hdf5::Group>
    where /* T: Deref<Target=hdf5::Group>, */
          S: Into<String>
{
    let n: String = name.into();
    if !parent.member_names()?.contains(&n) {
        parent.create_group(n.as_str())
    } else {
        parent.group(n.as_str())
    }
}

macro_rules! group {
    ($home:expr, $name:expr) => {
        util::get_or_create_group($home, $name)?
    };
    ($home:expr, $name:expr, $($rest:tt),*) => {
        group!(&util::get_or_create_group($home, $name)?, $($rest),*)
    };
}

macro_rules! dataset {
    ($home:expr, $name:expr) => {
        { let name: String = $name.into();
          $home.dataset(name.as_str())?.read()? }
    };
    ($home:expr, $name:expr, $($rest:tt),*) => {
        dataset!($home.group($name)?, $($rest),*)
    };
}

macro_rules! write_dataset {
    ($array:ident: $type:ty => $home:expr, $name:expr) => {
        { let name: String = $name.into();
          $home.new_dataset::<$type>().shape($array.shape()).create(name.as_str())?.write($array.view())? }
    };
    ($array:ident: $type:ty => $home:expr, $name:expr, $($rest:tt),*) => {
        write_dataset!($array: $type => util::get_or_create_group($home, $name)?, $($rest),*)
    };
}

macro_rules! write_attribute {
    ($type:ty; $value:expr => $home:expr, $name:expr) => {
        { let name: String = $name.into();
          $home.new_attr::<$type>().create(name.as_str())?.write_scalar($value)? }
    };
    ($type:ty; $value:expr => $home:expr, $name:expr, $($rest:tt),*) => {
        write_attribute!($type; $value => util::get_or_create_group($home, $name)?, $($rest),*)
    };
}

