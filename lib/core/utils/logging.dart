String truncateForLog(String value, [int maxChars = 500]) {
  if (value.length <= maxChars) {
    return value;
  }
  return '${value.substring(0, maxChars)}…';
}
