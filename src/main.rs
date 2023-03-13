use std::collections::{HashMap, HashSet};
struct Solution;

// Definition for singly-linked list. Used in LEETCODE 2.
#[derive(PartialEq, Eq, Clone, Debug)]
pub struct ListNode {
    pub val: i32,
    pub next: Option<Box<ListNode>>,
}

impl ListNode {
    #[inline]
    fn new(val: i32) -> Self {
        ListNode { next: None, val }
    }
}

pub struct MaxHeap {
    pub nums: Vec<i32>,
}

impl MaxHeap {
    fn new(nums: Vec<i32>) -> MaxHeap {
        let mut heap = MaxHeap { nums };
        heap = heap.CreateHeap();
        heap.Sort()
    }

    fn CreateHeap(mut self) -> MaxHeap {
        let size = self.nums.len();
        for i in (0..size).rev() {
            self = self.MaxHeap(size, i);
        }
        self
    }

    fn MaxHeap(mut self, heap_size: usize, i: usize) -> MaxHeap {
        let mut largest = i;
        let left = 2 * i + 1;
        let right = 2 * i + 2;
        if left < heap_size && self.nums[left] > self.nums[largest] {
            largest = left;
        }
        if right < heap_size && self.nums[right] > self.nums[largest] {
            largest = right;
        }
        if largest != i {
            self.nums.swap(i, largest);
            return self.MaxHeap(heap_size, largest);
        }
        self
    }

    fn Sort(mut self) -> MaxHeap {
        let mut size = self.nums.len();
        for i in (0..size).rev() {
            self.nums.swap(0, i);
            size -= 1;
            self = self.MaxHeap(size, 0);
        }
        self
    }
}

impl Solution {
    /**
     * LEETCODE QUESTIONS
     **/
    /**
     * LEETCODE 1. Two Sum [https://leetcode.com/problems/two-sum/] (Easy)
     *
     * Description:
     *   Given an array of integers and an integer target, return indices of two numbers that up to the target. May assume that each target has a single solution in the array, no index duplicates.
     *
     * Implementation:
     *   Theres a simple iterate through i and j, but that's O(n^2) and we can do better.
     *   1. Create a hashmap of the values and their index
     *   2. Iterate through the array, for each element, check if the target - element is in the hashmap
     *   3. If it is, return the index of the element and the index of the target - element
     *   4. If not, add the element to the hashmap
     *   5. If we get to the end of the array and don't find a match, return an empty vector
     *   This uses more memory than just the array, but i think it's much nicer to work with. as it works with the given assumptions (one and only one solution) and is O(n) time.
     *   
     * Notes:
     *   - Interesting Algorithm: Two Pointer
     *       This is almost two pointer problem, You could sort the array and then make use of that algorithm. Essentially, since we're sorted
     *       we can start at the beginning and end of the array and check if the sum is greater than or less than the target. If it's greater than the target, we can decrement the end pointer
     *       if it's less than the target we can increment the start pointer. This is O(nlogn) + O(n) = O(nlogn) which is better than O(n^2)
     */
    pub fn two_sum(nums: Vec<i32>, target: i32) -> Vec<i32> {
        let mut map: HashMap<i32, i32> = HashMap::new();
        for (i, num) in nums.iter().enumerate() {
            let found = map.get(&(target - num));
            if found.is_some() {
                return vec![*found.unwrap(), i as i32];
            }
            map.insert(*num, i as i32);
        }
        vec![]
    }

    /**
     * LEETCODE 2. Add Two Numbers [https://leetcode.com/problems/add-two-numbers/] (Medium)
     *
     * Description:
     *   Given two non-empty linked lists representing two non-negative integers, add the two numbers and return it as a linked list.
     *   You may assume the two numbers do not contain any leading zero, except the number 0 itself.
     *   The digits are stored in reverse order, and each of their nodes contains a single digit. Add the two numbers and return the sum as a linked list.
     *
     * Implementation:
     *   This is a recursive implementation, I don't love the readability here. I think it's a bit messy and could be cleaned up a bit. Steps taken are:
     *   1. Unwrap the first and second list nodes, if they are None, set them to 0
     *   2. Create a new ListNode with the sum of the two values mod 10 (to get the ones place)
     *   3. Create a carry variable, which is the sum of the two values divided by 10 (to get the tens place)
     *   4. If both the first and second next are None, we're at the end of the list. If the carry is 1, we need to add a new ListNode with a value of 1
     *   5. If the first next is None, but the second next is not, we need to add the carry to the second next value and recurse
     *   6. If the second next is None, but the first next is not, we need to add the carry to the first next value and recurse
     *   7. If both the first and second next are not None, we need to add the carry to the first next value and recurse
     *   8. Return the ListNode
     *
     * Cleaned up Implementation
     *   1. Unwrap the first and second list nodes, if they are None, set them to 0
     *   2. Create a new ListNode with the sum of the two values mod 10 (to get the ones place)
     *   3. Create a carry variable, which is the sum of the two values divided by 10 (to get the tens place) should be 0 or 1
     *   4. If both the first and second next are None and carry = 0 we're done!
     *   5. We then unwrap_or default to 0 the first and second next values, and add the carry to l1.next.val (this cuts out a lot of branching logic)
     *   6. recurse, our ListNode.next and return the ListNode Boxxed up.
     *   
     * Notes:
     *  - Interesting Algorithm: Recursion
     *      Not sure I really call recursion an algorithm, But I guess so, more of a methodology to me.
     */
    pub fn add_two_numbers(
        l1: Option<Box<ListNode>>,
        l2: Option<Box<ListNode>>,
    ) -> Option<Box<ListNode>> {
        // First solution here. Very messy, lots of "branching logic" for almost the same thing.
        /*  let first = l1.unwrap_or(Box::new(ListNode::new(0)));
        let second = l2.unwrap_or(Box::new(ListNode::new(0)));
        let mut returning = ListNode::new((first.val + second.val) % 10);
        let carry = (first.val + second.val) / 10;
        if let None = first.next {
            if let None = second.next {
                if carry == 1 {
                    returning.next = Some(Box::new(ListNode::new(1)));
                }
                return Some(Box::new(returning));
            }
            if let Some(next2) = &second.next {
                let mut l2_next = next2.clone();
                l2_next.val = l2_next.val + carry;
                returning.next = Solution::add_two_numbers(None, Some(l2_next));
            }
        }
        if let Some(next) = &first.next {
            if let None = second.next {
                let mut l1_next = next.clone();
                l1_next.val = l1_next.val + carry;
                returning.next = Solution::add_two_numbers(Some(l1_next), None);
            }
            if let Some(next2) = &second.next {
                let mut l1_next = next.clone();
                l1_next.val = l1_next.val + carry;
                returning.next = Solution::add_two_numbers(Some(l1_next), Some(next2.clone()));
            }
        }
        Some(Box::new(returning)) */
        // Second better solution, much cleaner and faster.
        let first = l1.unwrap_or(Box::new(ListNode::new(0)));
        let second = l2.unwrap_or(Box::new(ListNode::new(0)));
        let mut returning = ListNode::new((first.val + second.val) % 10);
        let carry = (first.val + second.val) / 10;
        let either_next = first.next.is_some() || second.next.is_some();
        if !either_next && carry == 0 {
            return Some(Box::new(returning));
        }
        let mut l1_next = first.next.unwrap_or(Box::new(ListNode::new(0)));
        let l2_next = second.next.unwrap_or(Box::new(ListNode::new(0)));
        l1_next.val = l1_next.val + carry;

        returning.next = Solution::add_two_numbers(Some(l1_next), Some(l2_next));

        Some(Box::new(returning))
    }

    /**
     * LEETCODE 3. Longest Substring Without Repeating Characters [https://leetcode.com/problems/longest-substring-without-repeating-characters/] (Medium)
     *
     * Description:
     *  Given a string, find the length of the longest substring without repeating characters.
     *
     * Implementation:
     *   Honestly somewhat impressed with Github Copilot on this one. It got the basic idea right, but I had to clean it up a bit. Implementation provided primarily by copilot, added a max + 1 since it's implementation was always off by 1.
     *   1. Create a HashMap to store the characters and their index
     *   2. Create a start variable to keep track of the start of the substring
     *   3. Create a max variable to keep track of the max length of the substring
     *   4. Iterate through the string
     *   5. If the character is in the HashMap, and the index is greater than the start, set the start to the index + 1
     *   6. If the index - start is greater than the max, set the max to the index - start
     *   7. Add the character and index to the HashMap
     *   8. Return the max
     *   Going to try my hand at a slightly better(?) implementation
     *   Tried using a HashSet implementation, worked for about 50% of cases and then I realized the necessity of the index. So I went back to the HashMap implementation.
     *   Only real difference in my implementation was the use of the max function instead of if statements.
     *
     * Notes:
     *   - Interesting Algorithm:
     */
    pub fn length_of_longest_substring(s: String) -> i32 {
        /* Copilot Wuz Here */
        /*         let mut map = HashMap::new();
        let mut start = 0;
        let mut max = 0;
        for (i, c) in s.chars().enumerate() {
            if let Some(index) = map.get(&c) {
                if *index >= start {
                    start = index + 1;
                }
            }
            max = max.max((i - start) + 1);
            map.insert(c, i);
        }
        max as i32 */
        let mut current = 0;
        let mut max = 0;
        let mut map = HashMap::new();
        for (i, c) in s.chars().enumerate() {
            if let Some(ind) = map.get(&c) {
                current = current.max(ind + 1);
            }
            println!("{:?}, {:?}", c, i);
            map.insert(c, i);
            max = max.max((i - current) + 1);
        }
        max as i32
    }

    /**
     * LEETCODE 4. Median of Two Sorted Arrays [https://leetcode.com/problems/median-of-two-sorted-arrays/] (Hard)
     *
     * Description:
     *   Given two sorted arrays nums1 and nums2 of size m and n respectively, return the median of the two sorted arrays. Must be O(log(m+n)) Time complexity.
     *
     * Implementation:
     *   Just given there is a strict time complexity requirement, tells me theres a specific algorithm to use here. Binary Search seems somewhat reasonable for the problem. Hardest part being merging the two arrays or having a strategy to do something similar.
     *   1. Create a new array to store the merged arrays
     *   2. Create a pointer for each array
     *   3. While the pointers are less than the length of the array, compare the values at the pointers
     *   4. If the value at the first pointer is less than the value at the second pointer, push the value at the first pointer to the new array and increment the first pointer
     *   5. If the value at the second pointer is less than the value at the first pointer, push the value at the second pointer to the new array and increment the second pointer
     *   6. If the values are equal, push both values to the new array and increment both pointers
     *   7. Once the pointers are greater than the length of the array, push the remaining values to the new array
     *   8. If the length of the new array is odd, return the middle value
     *   9. If the length of the new array is even, return the average of the two middle values
     *
     * Notes:
     *   - Interesting Algorithm: Binary Search
     *     Very simple algorithm for a reasonable execution time. I'm sure there are more efficient ways to do this, but this is the first thing that came to mind.
     *   -
     */
    pub fn find_median_sorted_arrays(nums1: Vec<i32>, nums2: Vec<i32>) -> f64 {
        if nums1.len() < nums2.len() {
            return Solution::find_median_sorted_arrays(nums2, nums1);
        }

        let mut merged = Vec::new();
        let mut i = 0;
        let mut j = 0;

        let total_size = nums1.len() + nums2.len();
        while i < nums1.len() && j < nums2.len() {
            if nums1[i] < nums2[j] {
                merged.push(nums1[i]);
                i += 1;
            } else if nums2[j] < nums1[i] {
                merged.push(nums2[j]);
                j += 1;
            } else {
                merged.push(nums1[i]);
                merged.push(nums2[j]);
                i += 1;
                j += 1;
            }
        }

        if i < nums1.len() {
            while i + j < total_size / 2 {
                merged.push(nums1[i]);
                i += 1;
            }
        }
        if j < nums2.len() {
            while i + j < total_size / 2 {
                merged.push(nums2[j]);
                j += 1;
            }
        }

        if total_size % 2 == 0 {
            (merged[total_size / 2] + merged[(total_size / 2) - 1]) as f64 / 2.0
        } else {
            merged[total_size / 2] as f64
        }
    }

    /** LEETCODE 28. Find the Index of the First Occurence in a String [https://leetcode.com/problems/find-the-index-of-the-first-occurrence-in-a-string/] (Medium)
     *
     * Description:
     *   Given a needle, find the first occurrence of the needle within a haystack, if it exists. Return -1 if it doesn't exist.
     *
     * Implementation:
     *   This is a pretty simple solution.
     *   1. start at 0, make sure the needle + index is not longer than the haystack
     *   2. check if the needle is at the current index. If it is, return the index.
     *   3. If not, increment the index and try again.
     *
     * Notes:
     *   - Interesting Algorithm: KMP
     *     There is apparently a KMP algorithm, don't think it's really what I want in this scenario but keep it in mind for similar problems where I need to get more than 1 solution.
     *     Looking more into KMP it seems you don't take a fresh new slice, instead look only at the first slice, then compare the next character to the final character of your slice.
     *     Not quite that simple, but along that lines, you can only do that when you know you've already met the previous character conditions required. For this problem KPM seems overkill...
     **/
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

    /**
     * LEETCODE 53. Maximum Subarray [https://leetcode.com/problems/maximum-subarray/?envType=study-plan&id=data-structure-i] (Medium)
     *
     * Description:
     *   Given an integer array nums, find the contiguous subarray (containing at least one number) which has the largest sum and return its sum.
     *
     * Implementation:
     *   This one took me a second to understand.
     *     1. Start with the first element as the max and current sum.
     *     2. Loop through the array, starting at the second element.
     *     3. We either take the current element or the current element + the current sum, this works because with the max of the two we are either "continuing" the array, or starting our new contiguous array
     *     4. Finall take the max of current and max.
     *
     * Notes:
     *   - Interesting Algorithm: Kadane's Algorithm
     *       This is basically what is being done here, but I didn't know it had a name
     */
    pub fn max_sub_array(nums: Vec<i32>) -> i32 {
        let mut max = nums[0];
        let mut current = nums[0];
        for i in 1..nums.len() {
            current = nums[i].max(current + nums[i]);
            max = max.max(current);
        }
        max
    }

    /**Leet code 217. Contains Duplicate [https://leetcode.com/problems/contains-duplicate/?envType=study-plan&id=data-structure-i] (Easy)
     * I used a hash map as an easy way to keep track of if something has been seen before. You could also sort the array and then check if the next element is the same as the current one.
     **/
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

    /** LEETCODE 443. String Compression [https://leetcode.com/problems/string-compression/] (Medium)
     *
     */
    pub fn compress(chars: &mut Vec<char>) -> i32 {
        let mut counts: Vec<i32> = Vec::new();
        let mut started = true;
        let mut current: char = chars[0];
        let mut i = 0;
        let smaller_iter = chars
            .into_iter()
            .filter(|val| {
                if started {
                    started = false;
                    return true;
                }

                if **val == current {
                    if i >= counts.len() {
                        counts.push(2);
                    } else {
                        counts[i] += 1;
                    }
                    return false;
                }
                current = **val;
                i += 1;
                if i >= counts.len() {
                    counts.push(1);
                }
                return true;
            })
            .map(|val| {
                let return_val = val.to_string().chars().next().unwrap();
                return_val
            })
            .collect::<Vec<char>>();
        let mut return_vec: Vec<char> = Vec::new();
        for (i, string) in smaller_iter.iter().enumerate() {
            let count = counts.get(i).unwrap_or(&1);
            return_vec.push(*string);
            if i >= counts.len() {
                continue;
            }
            if counts[i] > 1 {
                for str_count in counts[i].to_string().chars().into_iter() {
                    return_vec.push(str_count);
                }
            }
        }
        *chars = return_vec;
        chars.len() as i32
    }

    /** LEETCODE 912. Sort an Array [https://leetcode.com/problems/sort-an-array/] (Medium)
     *
     */
    pub fn sort_array(nums: Vec<i32>) -> Vec<i32> {
        let heap = MaxHeap::new(nums);
        return heap.nums;
    }
}

pub fn main() {}
