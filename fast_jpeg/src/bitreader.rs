/// Reads individual bits from one entropy-coded scan segment, transparently
/// undoing byte stuffing (0xFF 0x00 -> a literal 0xFF data byte). The slice
/// handed to this reader is assumed to already have any trailing restart
/// marker excluded by the caller.
pub struct BitReader<'a> {
    data: &'a [u8],
    pos: usize,
    bit_buffer: u32,
    bit_count: u32,
}

impl<'a> BitReader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        BitReader {
            data,
            pos: 0,
            bit_buffer: 0,
            bit_count: 0,
        }
    }

    fn fill(&mut self) {
        while self.bit_count <= 24 {
            if self.pos >= self.data.len() {
                // Ran out of real data: pad with zero bits. A well-formed
                // segment has exactly enough bits for its MCUs, so this only
                // kicks in right at the tail end.
                self.bit_buffer |= 0u32 << (24 - self.bit_count);
                self.bit_count += 8;
                continue;
            }

            let byte = self.data[self.pos];
            if byte == 0xFF {
                if self.pos + 1 < self.data.len() && self.data[self.pos + 1] == 0x00 {
                    self.pos += 2; // stuffed 0xFF, consume the stuffing byte
                } else {
                    // A real marker shouldn't appear inside a segment we've
                    // already bounded correctly; treat as end-of-data rather
                    // than misinterpreting marker bytes as entropy data.
                    self.pos = self.data.len();
                    continue;
                }
                self.bit_buffer |= (byte as u32) << (24 - self.bit_count);
                self.bit_count += 8;
            } else {
                self.bit_buffer |= (byte as u32) << (24 - self.bit_count);
                self.bit_count += 8;
                self.pos += 1;
            }
        }
    }

    pub fn get_bit(&mut self) -> u32 {
        self.fill();
        let bit = self.bit_buffer >> 31;
        self.bit_buffer <<= 1;
        self.bit_count = self.bit_count.saturating_sub(1);
        bit
    }

    pub fn get_bits(&mut self, n: u8) -> u32 {
        let mut value = 0u32;
        for _ in 0..n {
            value = (value << 1) | self.get_bit();
        }
        value
    }

    /// The JPEG "EXTEND" procedure (spec section F.2.2.1): turns a decoded
    /// magnitude-category value back into a signed difference/coefficient.
    pub fn receive_extend(&mut self, s: u8) -> i32 {
        if s == 0 {
            return 0;
        }
        let v = self.get_bits(s) as i32;
        if v < (1 << (s - 1)) {
            v - (1 << s) + 1
        } else {
            v
        }
    }
}
