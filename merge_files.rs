use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, Write};

fn main() -> io::Result<()> {
    let paths = fs::read_dir("./data/Lenta/texts").unwrap();

    let output_path = "./data/Lenta/sentences.txt";
    let mut output = File::create(output_path)?;

    for path in paths {
        let input = File::open(path.unwrap().path()).unwrap();
        let buffered = BufReader::new(input);

        for line in buffered.lines().skip(1) {
            let record = line?;

            for sentence in record.split(".").filter(|x| !x.is_empty()) {
                writeln!(output, "{}", sentence.trim())?;
            }
        }

    }

    Ok(())
}
