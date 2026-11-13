use crate::bitreader::BitReader;
use crate::DecodeError;

/// A JPEG Huffman table, decoded via the classic mincode/maxcode/valptr
/// method described in the JPEG spec (Annex F, Figures F.15-F.18) — read one
/// bit at a time, and at each length check whether the code built so far
/// falls within that length's assigned range.
#[derive(Clone)]
pub struct HuffmanTable {
    mincode: [i32; 17],
    maxcode: [i32; 17],
    valptr: [i32; 17],
    huffval: Vec<u8>,
}

impl HuffmanTable {
    /// `counts[i]` is how many codes of length `i + 1` exist; `symbols` is
    /// the flat list of symbol values in canonical order (shortest codes,
    /// then longer ones), exactly as DHT segments store them.
    pub fn build(counts: &[u8; 16], symbols: &[u8]) -> Self {
        let mut huffsize = Vec::new();
        for (len_idx, &count) in counts.iter().enumerate() {
            let length = (len_idx + 1) as u8;
            for _ in 0..count {
                huffsize.push(length);
            }
        }

        let mut huffcode = vec![0u32; huffsize.len()];
        let mut code = 0u32;
        let mut size_index = 0usize;
        while size_index < huffsize.len() {
            let si = huffsize[size_index];
            while size_index < huffsize.len() && huffsize[size_index] == si {
                huffcode[size_index] = code;
                code += 1;
                size_index += 1;
            }
            code <<= 1;
        }

        let mut mincode = [0i32; 17];
        let mut maxcode = [-1i32; 17];
        let mut valptr = [0i32; 17];
        let mut p = 0usize;
        for l in 1..=16usize {
            if counts[l - 1] > 0 {
                valptr[l] = p as i32;
                mincode[l] = huffcode[p] as i32;
                p += counts[l - 1] as usize;
                maxcode[l] = huffcode[p - 1] as i32;
            }
        }

        HuffmanTable {
            mincode,
            maxcode,
            valptr,
            huffval: symbols.to_vec(),
        }
    }

    pub fn decode(&self, reader: &mut BitReader) -> Result<u8, DecodeError> {
        let mut code = reader.get_bit() as i32;
        let mut len = 1usize;
        while len <= 16 {
            if self.maxcode[len] != -1 && code <= self.maxcode[len] {
                let index = (self.valptr[len] + (code - self.mincode[len])) as usize;
                return self
                    .huffval
                    .get(index)
                    .copied()
                    .ok_or(DecodeError::Malformed("huffman symbol index out of range"));
            }
            code = (code << 1) | reader.get_bit() as i32;
            len += 1;
        }
        Err(DecodeError::Malformed("no huffman code matched (corrupt scan data)"))
    }
}
