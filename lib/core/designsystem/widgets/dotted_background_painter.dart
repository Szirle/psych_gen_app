import 'package:flutter/material.dart';

class DottedBackgroundPainter extends CustomPainter {
  const DottedBackgroundPainter({
    required this.color,
    required this.spacing,
    this.origin = Offset.zero,
    this.radius = 1.8,
    this.overflow = 2000,
  });

  final Color color;
  final double spacing;
  final Offset origin;
  final double radius;

  /// Extra pixels beyond the widget bounds to paint dots into.
  /// This prevents dots from ending abruptly when the user pans or zooms out.
  final double overflow;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 1;

    final startX = (origin.dx % spacing) - overflow;
    final startY = (origin.dy % spacing) - overflow;
    final endX = size.width + overflow;
    final endY = size.height + overflow;

    for (double i = startX; i < endX; i += spacing) {
      for (double j = startY; j < endY; j += spacing) {
        canvas.drawCircle(Offset(i, j), radius, paint);
      }
    }
  }

  @override
  bool shouldRepaint(covariant DottedBackgroundPainter oldDelegate) =>
      color != oldDelegate.color ||
      spacing != oldDelegate.spacing ||
      origin != oldDelegate.origin ||
      radius != oldDelegate.radius ||
      overflow != oldDelegate.overflow;
}
