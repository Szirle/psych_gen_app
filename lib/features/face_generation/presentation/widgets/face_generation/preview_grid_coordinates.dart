/// Converts a top-to-bottom Flutter row index into an attribute level index.
///
/// Attribute levels are ordered low-to-high, but screen rows are laid out
/// top-to-bottom. Reversing the row makes higher values appear upward.
int verticalLevelForDisplayRow(int row, int levelCount) {
  if (row < 0 || row >= levelCount) {
    throw RangeError.range(row, 0, levelCount - 1, 'row');
  }
  return levelCount - 1 - row;
}
