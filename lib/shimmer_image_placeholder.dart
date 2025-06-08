import 'package:flutter/material.dart';
import 'package:shimmer/shimmer.dart';

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
          // Grid layout for 2D or 3D
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
          // Single row layout for 1D
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

class AnimatedImageWidget extends StatefulWidget {
  final Widget child;
  final Duration duration;

  const AnimatedImageWidget({
    super.key,
    required this.child,
    this.duration = const Duration(milliseconds: 500),
  });

  @override
  State<AnimatedImageWidget> createState() => _AnimatedImageWidgetState();
}

class _AnimatedImageWidgetState extends State<AnimatedImageWidget>
    with SingleTickerProviderStateMixin {
  late AnimationController _controller;
  late Animation<double> _fadeAnimation;
  late Animation<double> _scaleAnimation;

  @override
  void initState() {
    super.initState();
    _controller = AnimationController(
      duration: widget.duration,
      vsync: this,
    );

    _fadeAnimation = Tween<double>(
      begin: 0.0,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.easeInOut,
    ));

    _scaleAnimation = Tween<double>(
      begin: 0.8,
      end: 1.0,
    ).animate(CurvedAnimation(
      parent: _controller,
      curve: Curves.elasticOut,
    ));

    _controller.forward();
  }

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AnimatedBuilder(
      animation: _controller,
      builder: (context, child) {
        return Transform.scale(
          scale: _scaleAnimation.value,
          child: Opacity(
            opacity: _fadeAnimation.value,
            child: widget.child,
          ),
        );
      },
    );
  }
}
