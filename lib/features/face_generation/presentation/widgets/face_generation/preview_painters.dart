import 'dart:math' as math;
import 'dart:ui' as ui;
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/manipulated_dimension.dart';

class DottedBorderPainter extends CustomPainter {
  final Color color;
  final double strokeWidth;
  final double gap;
  final double dashWidth;
  final Radius radius;

  DottedBorderPainter({
    this.color = Colors.black,
    this.strokeWidth = 1.0,
    this.gap = 5.0,
    this.dashWidth = 5.0,
    this.radius = const Radius.circular(0),
  });

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = strokeWidth
      ..style = PaintingStyle.stroke;

    Path path = Path();
    path.addRRect(RRect.fromRectAndRadius(
        Rect.fromLTWH(0, 0, size.width, size.height), radius));

    ui.PathMetrics pathMetrics = path.computeMetrics();
    for (ui.PathMetric pathMetric in pathMetrics) {
      double distance = 0.0;
      while (distance < pathMetric.length) {
        canvas.drawPath(
          pathMetric.extractPath(distance, distance + dashWidth),
          paint,
        );
        distance += dashWidth + gap;
      }
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}

class AxisWrappingPainter extends CustomPainter {
  final ManipulatedDimension? xDim;
  final ManipulatedDimension? yDim;
  final ManipulatedDimension? zDim;
  final Map<ManipulatedDimension, Color> dimensionColors;

  AxisWrappingPainter({
    required this.xDim,
    required this.yDim,
    required this.zDim,
    required this.dimensionColors,
  });

  static const double _arrowThickness = 2.5;
  static const double _arrowHeadSize = 10.0;

  @override
  void paint(Canvas canvas, Size size) {
    final double left = 0;
    final double right = size.width;
    final double bottom = size.height;
    final double top = 0;

    if (xDim != null) {
      _drawArrow(canvas,
          start: Offset(left, bottom),
          end: Offset(right, bottom),
          color: dimensionColors[xDim] ?? const Color(0xFF4A90E2));
      _drawLabel(
        canvas,
        text: xDim!.name.name,
        position: Offset((left + right) / 2, bottom + 14),
        color: dimensionColors[xDim] ?? const Color(0xFF4A90E2),
        rotateRadians: 0,
      );
    }

    if (yDim != null) {
      _drawArrow(canvas,
          start: Offset(left, bottom),
          end: Offset(left, top),
          color: dimensionColors[yDim] ?? const Color(0xFFD53F8C));
      _drawLabel(
        canvas,
        text: yDim!.name.name,
        position: Offset(left - 18, (top + bottom) / 2),
        color: dimensionColors[yDim] ?? const Color(0xFFD53F8C),
        rotateRadians: -math.pi / 2,
      );
    }

    if (zDim != null) {
      final Offset start = Offset(right - 40, top + 8);
      final Offset end = Offset(right, top + 8 - 40);
      _drawArrow(canvas,
          start: start,
          end: end,
          color: dimensionColors[zDim] ?? const Color(0xFF3DBDBA));
      _drawLabel(
        canvas,
        text: zDim!.name.name,
        position: Offset(end.dx + 4, end.dy),
        color: dimensionColors[zDim] ?? const Color(0xFF3DBDBA),
        rotateRadians: 0,
      );
    }
  }

  void _drawArrow(Canvas canvas,
      {required Offset start, required Offset end, required Color color}) {
    final Paint paint = Paint()
      ..color = color
      ..strokeWidth = _arrowThickness
      ..style = PaintingStyle.stroke
      ..strokeCap = StrokeCap.round;

    canvas.drawLine(start, end, paint);

    final Offset direction = (end - start);
    final double length = direction.distance;
    if (length <= 0.001) return;
    final Offset unit = direction / length;
    final Offset perp = Offset(-unit.dy, unit.dx);
    final Offset headBase = end - unit * _arrowHeadSize;
    final Offset p1 = end;
    final Offset p2 = headBase + perp * (_arrowHeadSize * 0.6);
    final Offset p3 = headBase - perp * (_arrowHeadSize * 0.6);

    final Path head = Path()
      ..moveTo(p1.dx, p1.dy)
      ..lineTo(p2.dx, p2.dy)
      ..lineTo(p3.dx, p3.dy)
      ..close();

    final Paint headPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    canvas.drawPath(head, headPaint);
  }

  void _drawLabel(Canvas canvas,
      {required String text,
      required Offset position,
      required Color color,
      required double rotateRadians}) {
    final TextPainter textPainter = TextPainter(
      text: TextSpan(
        text: text,
        style: TextStyle(
          color: color,
          fontSize: 12,
          fontWeight: FontWeight.w600,
          fontFamily: 'WorkSans',
        ),
      ),
      textDirection: ui.TextDirection.ltr,
    )..layout();

    final Size ts = textPainter.size;
    canvas.save();
    canvas.translate(position.dx, position.dy);
    canvas.rotate(rotateRadians);
    textPainter.paint(canvas, Offset(-ts.width / 2, -ts.height / 2));
    canvas.restore();
  }

  @override
  bool shouldRepaint(covariant AxisWrappingPainter oldDelegate) {
    return oldDelegate.xDim != xDim ||
        oldDelegate.yDim != yDim ||
        oldDelegate.zDim != zDim ||
        !mapEquals(oldDelegate.dimensionColors, dimensionColors);
  }
}


