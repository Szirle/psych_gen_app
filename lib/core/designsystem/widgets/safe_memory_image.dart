import 'dart:typed_data';
import 'package:flutter/material.dart';

/// A widget that safely displays an image from memory bytes,
/// handling decoding errors gracefully.
class SafeMemoryImage extends StatelessWidget {
  final Uint8List imageBytes;
  final double? width;
  final double? height;
  final BoxFit fit;
  final Widget? errorWidget;

  const SafeMemoryImage({
    super.key,
    required this.imageBytes,
    this.width,
    this.height,
    this.fit = BoxFit.cover,
    this.errorWidget,
  });

  @override
  Widget build(BuildContext context) {
    try {
      return Image.memory(
        imageBytes,
        width: width,
        height: height,
        fit: fit,
        errorBuilder: (context, error, stackTrace) {
          return _buildErrorWidget();
        },
        frameBuilder: (context, child, frame, wasSynchronouslyLoaded) {
          if (frame == null) {
            return _buildErrorWidget();
          }
          return child;
        },
      );
    } catch (e) {
      return _buildErrorWidget();
    }
  }

  Widget _buildErrorWidget() {
    return errorWidget ??
        Container(
          width: width,
          height: height,
          decoration: BoxDecoration(
            color: Colors.grey[200],
            borderRadius: BorderRadius.circular(6),
          ),
          child: Icon(
            Icons.broken_image_outlined,
            size: (width != null && height != null)
                ? (width! < height! ? width! * 0.4 : height! * 0.4)
                : 24,
            color: Colors.grey[400],
          ),
        );
  }
}

