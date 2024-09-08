import 'package:flutter/material.dart';

class DottedBackgroundPainter extends CustomPainter {
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.grey
      ..strokeWidth = 1;

    var dotSize = 2.0; // Size of the dots
    var spaceBetween = 50.0; // Space between dots

    for (double i = 0; i < size.width; i += spaceBetween) {
      for (double j = 0; j < size.height; j += spaceBetween) {
        canvas.drawCircle(Offset(i, j), dotSize, paint);
      }
    }
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => false;
}