import 'package:flutter/material.dart';
import 'package:psych_gen_app/constants/distributions.dart';
import 'package:psych_gen_app/model/manipulated_dimension.dart';

class DistributionRangeSelector extends StatelessWidget {
  final ManipulatedDimension dimension;
  final Color accentColor;
  final void Function(double start, double end) onRangeChanged;

  const DistributionRangeSelector({
    Key? key,
    required this.dimension,
    required this.accentColor,
    required this.onRangeChanged,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final distribution = normalDistributions[dimension.name] ??
        generateNormalDistributionPoints();

    const double graphHeight = 110;
    const double sliderHeight = 36;

    return LayoutBuilder(builder: (context, constraints) {
      final width = constraints.maxWidth;
      final startX = (dimension.rangeStart.clamp(0.0, 1.0)) * width;
      final endX = (dimension.rangeEnd.clamp(0.0, 1.0)) * width;

      return SizedBox(
        height: graphHeight + sliderHeight,
        width: double.infinity,
        child: Stack(
          clipBehavior: Clip.none,
          children: [
            // Graph
            Positioned(
              left: 0,
              right: 0,
              top: 0,
              height: graphHeight,
              child: CustomPaint(
                painter: _DistributionPainter(
                  values: distribution,
                  highlightStart: dimension.rangeStart,
                  highlightEnd: dimension.rangeEnd,
                  color: accentColor,
                ),
              ),
            ),
            // Slider flush with graph bottom, track fills full width
            Positioned(
              left: 0,
              right: 0,
              top: graphHeight - 19, // slight overlap to remove visual gap
              height: sliderHeight,
              child: SliderTheme(
                data: SliderTheme.of(context).copyWith(
                  rangeTrackShape: const _FullWidthRangeSliderTrackShape(),
                  activeTrackColor: accentColor,
                  inactiveTrackColor: Colors.black12,
                  thumbColor: accentColor
                ),
                child: RangeSlider(
                  values: RangeValues(dimension.rangeStart, dimension.rangeEnd),
                  min: 0,
                  max: 1,
                  divisions: 100,
                  onChanged: (values) {
                    onRangeChanged(values.start, values.end);
                  },
                ),
              ),
            ),
            // Value bubbles under thumbs (0..100)
            Positioned(
              left: startX - 10,
              top: graphHeight + sliderHeight - 30,
              child: _ValueBubble(
                value: (dimension.rangeStart * 100).round(),
                color: accentColor,
              ),
            ),
            Positioned(
              left: endX - 16,
              top: graphHeight + sliderHeight - 30,
              child: _ValueBubble(
                value: (dimension.rangeEnd * 100).round(),
                color: accentColor,
              ),
            ),
          ],
        ),
      );
    });
  }
}

class _DistributionPainter extends CustomPainter {
  final List<double> values;
  final double highlightStart;
  final double highlightEnd;
  final Color color;

  _DistributionPainter({
    required this.values,
    required this.highlightStart,
    required this.highlightEnd,
    required this.color,
  });

  @override
  void paint(Canvas canvas, Size size) {
    if (values.isEmpty) return;

    final strokePaint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2.0;

    final fillPaint = Paint()
      ..color = color.withOpacity(0.2)
      ..style = PaintingStyle.fill;

    final path = Path();
    final highlightedPath = Path();

    final count = values.length;
    final dx = size.width / (count - 1);
    final maxVal = 1.0; // already normalized to 1

    // Build full curve path
    for (int i = 0; i < count; i++) {
      final x = i * dx;
      final y = size.height - (values[i] / maxVal) * size.height;
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    // Build highlighted fill path under the curve in the selected range
    final startPx = (highlightStart.clamp(0.0, 1.0)) * size.width;
    final endPx = (highlightEnd.clamp(0.0, 1.0)) * size.width;
    final lowPx = startPx < endPx ? startPx : endPx;
    final highPx = startPx < endPx ? endPx : startPx;

    // We approximate integration by sampling the same values and clipping to range
    bool started = false;
    for (int i = 0; i < count; i++) {
      final x = i * dx;
      if (x < lowPx) continue;
      if (x > highPx) break;
      final y = size.height - (values[i] / maxVal) * size.height;
      if (!started) {
        highlightedPath.moveTo(x, size.height);
        highlightedPath.lineTo(x, y);
        started = true;
      } else {
        highlightedPath.lineTo(x, y);
      }
    }
    if (started) {
      highlightedPath.lineTo(highPx, size.height);
      highlightedPath.close();
      canvas.drawPath(highlightedPath, fillPaint);
    }

    // Draw curve on top
    canvas.drawPath(path, strokePaint);
  }

  @override
  bool shouldRepaint(covariant _DistributionPainter oldDelegate) {
    return oldDelegate.values != values ||
        oldDelegate.highlightStart != highlightStart ||
        oldDelegate.highlightEnd != highlightEnd ||
        oldDelegate.color != color;
  }
}

class _FullWidthRangeSliderTrackShape extends RoundedRectRangeSliderTrackShape {
  const _FullWidthRangeSliderTrackShape();

  @override
  Rect getPreferredRect({
    required RenderBox parentBox,
    Offset offset = Offset.zero,
    required SliderThemeData sliderTheme,
    bool isEnabled = false,
    bool isDiscrete = false,
  }) {
    final double trackHeight = sliderTheme.trackHeight ?? 4.0;
    final double trackLeft = offset.dx;
    final double trackTop =
        offset.dy + (parentBox.size.height - trackHeight) / 2;
    final double trackWidth = parentBox.size.width;
    return Rect.fromLTWH(trackLeft, trackTop, trackWidth, trackHeight);
  }
}

class _ValueBubble extends StatelessWidget {
  final int value;
  final Color color;

  const _ValueBubble({Key? key, required this.value, required this.color})
      : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(4),
        border: Border.all(color: color, width: 1),
      ),
      child: Text(
        value.toString(),
        style:
            TextStyle(color: color, fontSize: 10, fontWeight: FontWeight.w600),
      ),
    );
  }
}
