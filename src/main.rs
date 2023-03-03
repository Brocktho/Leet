use std::collections::HashMap;
struct Solution;

impl Solution {
    pub fn str_str(haystack: String, needle: String) -> i32 {
        let mut start = 0;
        loop {
            if start + needle.len() > haystack.len() {
                return -1;
            }
            let start_word = &haystack[start..start + needle.len()];
            if start_word == needle {
                return start as i32;
            }
            start += 1;
        }
    }
    pub fn contains_duplicate(nums: Vec<i32>) -> bool {
        let mut set: HashMap<i32, i32> = HashMap::new();
        for num in nums {
            let found = set.get(&num).unwrap_or(&0);
            if found > &0 {
                return true;
            }
            set.insert(num, 1);
        }
        false
    }
}

pub fn main() {}
