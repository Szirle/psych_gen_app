import 'package:flutter/material.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension.dart';

class DistributionRangeSelector extends StatelessWidget {
  final ManipulatedDimension dimension;
  final Color accentColor;
  final void Function(double start, double end) onRangeChanged;
  final List<double> values;
  final double? currentStart;
  final double? currentEnd;

  const DistributionRangeSelector({
    Key? key,
    required this.dimension,
    required this.accentColor,
    required this.onRangeChanged,
    required this.values,
    this.currentStart,
    this.currentEnd,
  }) : super(key: key);

  @override
  Widget build(BuildContext context) {
    final distribution =
        values.isNotEmpty ? values : List<double>.filled(100, 0);
    final startVal = (currentStart ?? dimension.rangeStart).clamp(0.0, 1.0);
    final endVal = (currentEnd ?? dimension.rangeEnd).clamp(0.0, 1.0);

    const double graphHeight = 110;
    const double sliderHeight = 36;

    return LayoutBuilder(builder: (context, constraints) {
      final width = constraints.maxWidth;
      final startX = (startVal) * width;
      final endX = (endVal) * width;

      return SizedBox(
        height: graphHeight + sliderHeight,
        width: double.infinity,
        child: Stack(
          clipBehavior: Clip.none,
          children: [
            Positioned(
              left: 0,
              right: 0,
              top: 0,
              height: graphHeight,
              child: CustomPaint(
                painter: _DistributionPainter(
                  values: distribution,
                  highlightStart: startVal,
                  highlightEnd: endVal,
                  color: accentColor,
                ),
              ),
            ),
            Positioned(
              left: 0,
              right: 0,
              top: graphHeight - 19,
              height: sliderHeight,
              child: SliderTheme(
                data: SliderTheme.of(context).copyWith(
                    rangeTrackShape: const _FullWidthRangeSliderTrackShape(),
                    activeTrackColor: accentColor,
                    inactiveTrackColor: Colors.black12,
                    thumbColor: accentColor),
                child: RangeSlider(
                  values: RangeValues(startVal, endVal),
                  min: 0,
                  max: 1,
                  divisions: 100,
                  onChanged: (values) {
                    onRangeChanged(values.start, values.end);
                  },
                ),
              ),
            ),
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
    final maxVal = 1.0;

    for (int i = 0; i < count; i++) {
      final x = i * dx;
      final y = size.height - (values[i] / maxVal) * size.height;
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }

    final startPx = (highlightStart.clamp(0.0, 1.0)) * size.width;
    final endPx = (highlightEnd.clamp(0.0, 1.0)) * size.width;
    final lowPx = startPx < endPx ? startPx : endPx;
    final highPx = startPx < endPx ? endPx : startPx;

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
