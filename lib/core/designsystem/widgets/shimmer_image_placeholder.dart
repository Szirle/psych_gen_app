import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:shimmer/shimmer.dart';
import 'package:psych_gen_app/core/designsystem/widgets/safe_memory_image.dart';

class AsyncGeneratedImageTile extends StatelessWidget {
  final Uint8List? imageBytes;
  final double size;
  final bool isLoading;
  final double borderRadius;

  const AsyncGeneratedImageTile({
    super.key,
    required this.imageBytes,
    required this.size,
    required this.isLoading,
    this.borderRadius = 6,
  });

  @override
  Widget build(BuildContext context) {
    return Container(
      width: size,
      height: size,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(borderRadius),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.1),
            blurRadius: borderRadius > 6 ? 8 : 4,
            offset: Offset(0, borderRadius > 6 ? 4 : 2),
          ),
        ],
      ),
      child: ClipRRect(
        borderRadius: BorderRadius.circular(borderRadius),
        child: Stack(
          fit: StackFit.expand,
          children: [
            _GeneratedImagePlaceholder(size: size),
            if (imageBytes != null)
              _InPlaceImageReplacement(
                imageBytes: imageBytes!,
                size: size,
              ),
            if (isLoading) const _ShimmerLoadingOverlay(),
          ],
        ),
      ),
    );
  }
}

class _GeneratedImagePlaceholder extends StatelessWidget {
  final double size;

  const _GeneratedImagePlaceholder({required this.size});

  @override
  Widget build(BuildContext context) {
    return ColoredBox(
      color: Colors.grey.shade300,
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.image_outlined,
            size: (size * 0.25).clamp(16.0, 28.0),
            color: Colors.grey[500],
          ),
          SizedBox(height: size * 0.06),
          Container(
            width: size * 0.4,
            height: (size * 0.08).clamp(4.0, 8.0),
            decoration: BoxDecoration(
              color: Colors.grey[400],
              borderRadius: BorderRadius.circular(4),
            ),
          ),
          SizedBox(height: size * 0.03),
          Container(
            width: size * 0.6,
            height: (size * 0.06).clamp(3.0, 6.0),
            decoration: BoxDecoration(
              color: Colors.grey[400],
              borderRadius: BorderRadius.circular(3),
            ),
          ),
        ],
      ),
    );
  }
}

class _ShimmerLoadingOverlay extends StatelessWidget {
  const _ShimmerLoadingOverlay();

  @override
  Widget build(BuildContext context) {
    return IgnorePointer(
      child: Shimmer.fromColors(
        baseColor: Colors.white.withOpacity(0.04),
        highlightColor: Colors.white.withOpacity(0.48),
        period: const Duration(milliseconds: 1200),
        child: const ColoredBox(color: Colors.white),
      ),
    );
  }
}

class _InPlaceImageReplacement extends StatefulWidget {
  final Uint8List imageBytes;
  final double size;

  const _InPlaceImageReplacement({
    required this.imageBytes,
    required this.size,
  });

  @override
  State<_InPlaceImageReplacement> createState() =>
      _InPlaceImageReplacementState();
}

class _InPlaceImageReplacementState extends State<_InPlaceImageReplacement>
    with SingleTickerProviderStateMixin {
  late final AnimationController _controller;
  Uint8List? _outgoingBytes;
  bool _incomingReady = true;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 140),
      value: 1,
    );
  }

  @override
  void didUpdateWidget(covariant _InPlaceImageReplacement oldWidget) {
    super.didUpdateWidget(oldWidget);
    if (!listEquals(oldWidget.imageBytes, widget.imageBytes)) {
      _outgoingBytes = oldWidget.imageBytes;
      _incomingReady = false;
      _controller.value = 0;
    }
  }

  void _handleIncomingReady(Uint8List bytes) {
    if (_incomingReady ||
        !mounted ||
        !listEquals(bytes, widget.imageBytes)) {
      return;
    }
    _incomingReady = true;
    _controller.forward();
  }

  Widget _image(Uint8List bytes, {bool incoming = false}) {
    return SafeMemoryImage(
      imageBytes: bytes,
      width: widget.size,
      height: widget.size,
      fit: BoxFit.cover,
      onImageReady: incoming ? () => _handleIncomingReady(bytes) : null,
    );
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Stack(
          fit: StackFit.expand,
          children: [
            if (_outgoingBytes != null) _image(_outgoingBytes!),
            ClipRect(
              child: Align(
                alignment: Alignment.centerLeft,
                widthFactor: _controller.value,
                child: SizedBox.square(
                  dimension: widget.size,
                  child: _image(widget.imageBytes, incoming: true),
                ),
              ),
            ),
          ],
        );
      },
    );
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }
}

class ShimmerImagePlaceholder extends StatelessWidget {
  final double? width;
  final double? height;
  final int count;
  final int? rows;
  final int? cols;

  const ShimmerImagePlaceholder({
    super.key,
    this.width,
    this.height,
    this.count = 3,
    this.rows,
    this.cols,
  });

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final availableWidth = constraints.maxWidth;
        final availableHeight = constraints.maxHeight;

        final bool shouldUseGrid = rows != null && cols != null;

        if (shouldUseGrid) {
          final gridRows = rows!;
          final gridCols = cols!;
          final padding = 16.0;
          final itemPadding = 4.0;

          final availableImageWidth =
              (availableWidth - padding - (itemPadding * 2 * gridCols)) /
                  gridCols;
          final availableImageHeight =
              (availableHeight - padding - (itemPadding * 2 * gridRows)) /
                  gridRows;
          final imageSize = (availableImageWidth < availableImageHeight
                  ? availableImageWidth
                  : availableImageHeight)
              .clamp(30.0, 150.0);

          return Padding(
            padding: const EdgeInsets.all(8.0),
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: List.generate(gridRows, (row) {
                return Row(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: List.generate(gridCols, (col) {
                    return Padding(
                      padding: EdgeInsets.all(itemPadding),
                      child: Shimmer.fromColors(
                        baseColor: Colors.grey[300]!,
                        highlightColor: Colors.grey[100]!,
                        period: const Duration(milliseconds: 1200),
                        child: Container(
                          width: imageSize,
                          height: imageSize,
                          decoration: BoxDecoration(
                            color: Colors.grey[300],
                            borderRadius: BorderRadius.circular(6),
                            boxShadow: [
                              BoxShadow(
                                color: Colors.black.withOpacity(0.1),
                                blurRadius: 4,
                                offset: const Offset(0, 2),
                              ),
                            ],
                          ),
                          child: Column(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              Icon(
                                Icons.image_outlined,
                                size: (imageSize * 0.25).clamp(16.0, 24.0),
                                color: Colors.grey[400],
                              ),
                              SizedBox(height: imageSize * 0.06),
                              Container(
                                width: imageSize * 0.4,
                                height: (imageSize * 0.08).clamp(4.0, 8.0),
                                decoration: BoxDecoration(
                                  color: Colors.grey[400],
                                  borderRadius: BorderRadius.circular(4),
                                ),
                              ),
                              SizedBox(height: imageSize * 0.03),
                              Container(
                                width: imageSize * 0.6,
                                height: (imageSize * 0.06).clamp(3.0, 6.0),
                                decoration: BoxDecoration(
                                  color: Colors.grey[400],
                                  borderRadius: BorderRadius.circular(3),
                                ),
                              ),
                            ],
                          ),
                        ),
                      ),
                    );
                  }),
                );
              }),
            ),
          );
        } else {
          final padding = 16.0;
          final itemPadding = 8.0 * 2;
          final totalPadding = padding + (itemPadding * count);
          final availableImageWidth = availableWidth - totalPadding;
          final calculatedImageWidth = availableImageWidth / count;
          final imageWidth = calculatedImageWidth.clamp(20.0, 200.0);
          final imageHeight = height ?? imageWidth;

          return Padding(
            padding: const EdgeInsets.symmetric(horizontal: 8.0),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: List.generate(
                count,
                (index) => Flexible(
                  child: Padding(
                    padding: const EdgeInsets.all(8.0),
                    child: Shimmer.fromColors(
                      baseColor: Colors.grey[300]!,
                      highlightColor: Colors.grey[100]!,
                      period: const Duration(milliseconds: 1200),
                      child: Container(
                        width: imageWidth,
                        height: imageHeight,
                        decoration: BoxDecoration(
                          color: Colors.grey[300],
                          borderRadius: BorderRadius.circular(8),
                          boxShadow: [
                            BoxShadow(
                              color: Colors.black.withOpacity(0.1),
                              blurRadius: 8,
                              offset: const Offset(0, 4),
                            ),
                          ],
                        ),
                        child: Column(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            Icon(
                              Icons.image_outlined,
                              size: (imageWidth * 0.2).clamp(24.0, 40.0),
                              color: Colors.grey[400],
                            ),
                            SizedBox(height: imageHeight * 0.04),
                            Container(
                              width: imageWidth * 0.4,
                              height: (imageHeight * 0.06).clamp(8.0, 12.0),
                              decoration: BoxDecoration(
                                color: Colors.grey[400],
                                borderRadius: BorderRadius.circular(6),
                              ),
                            ),
                            SizedBox(height: imageHeight * 0.02),
                            Container(
                              width: imageWidth * 0.6,
                              height: (imageHeight * 0.04).clamp(6.0, 8.0),
                              decoration: BoxDecoration(
                                color: Colors.grey[400],
                                borderRadius: BorderRadius.circular(4),
                              ),
                            ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ),
            ),
          );
        }
      },
    );
  }
}
