import 'dart:typed_data';

void downloadBytes(
  Uint8List bytes, {
  required String filename,
  required String mimeType,
}) {
  throw UnsupportedError('File downloads are available in the web app only.');
}
