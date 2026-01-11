use std::fs;
use std::path::Path;

fn test_encode(path: &Path) -> datatest_stable::Result<()> {
    // Input (.bin) is the path
    let bin_path = path;
    let txt_path = path.with_extension("txt");

    let bin_data = fs::read(bin_path)?;
    let expected_txt = fs::read_to_string(txt_path)?;

    let encoded = base32768::encode(&bin_data);

    assert_eq!(encoded, expected_txt, "Encoding failed for {:?}", bin_path);

    Ok(())
}

fn test_decode(path: &Path) -> datatest_stable::Result<()> {
    // Input (.txt) is the path
    let txt_path = path;
    let bin_path = path.with_extension("bin");

    let txt_data = fs::read_to_string(txt_path)?;
    let expected_bin = fs::read(bin_path)?;

    let decoded = base32768::decode(&txt_data)?;

    assert_eq!(decoded, expected_bin, "Decoding failed for {:?}", txt_path);

    Ok(())
}

datatest_stable::harness! {
    {
        test = test_encode,
        root = "test-resources/pairs",
        pattern = r"^.*\.bin$",
    },
    {
        test = test_decode,
        root = "test-resources/pairs",
        pattern = r"^.*\.txt$",
    }
}
