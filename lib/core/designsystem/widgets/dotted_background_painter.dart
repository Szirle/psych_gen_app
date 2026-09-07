import 'package:flutter/material.dart';

class DottedBackgroundPainter extends CustomPainter {
  const DottedBackgroundPainter({
    required this.color,
    required this.spacing,
    this.origin = Offset.zero,
    this.radius = 1.8,
  });

  final Color color;
  final double spacing;
  final Offset origin;
  final double radius;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 1;

    final startX = origin.dx % spacing;
    final startY = origin.dy % spacing;

    for (double i = startX; i < size.width; i += spacing) {
      for (double j = startY; j < size.height; j += spacing) {
        canvas.drawCircle(Offset(i, j), radius, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant DottedBackgroundPainter oldDelegate) =>
      color != oldDelegate.color ||
      spacing != oldDelegate.spacing ||
      origin != oldDelegate.origin ||
      radius != oldDelegate.radius;
}
