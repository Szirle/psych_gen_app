import 'package:flutter/material.dart';

class AppTheme {
  static const Color _brand = Color(0xFF2B3A55);
  static const Color darkCanvas = Color(0xFF050505);
  static const Color lightCanvas = Color(0xFFF7F7F8);

  static ThemeData light() => _build(Brightness.light);

  static ThemeData dark() => _build(Brightness.dark);

  static ThemeData _build(Brightness brightness) {
    final isDark = brightness == Brightness.dark;
    final scheme = ColorScheme.fromSeed(
      seedColor: _brand,
      brightness: brightness,
      surface: isDark ? const Color(0xFF161719) : Colors.white,
    );

    return ThemeData(
      brightness: brightness,
      fontFamily: 'WorkSans',
      useMaterial3: true,
      colorScheme: scheme,
      scaffoldBackgroundColor:
          isDark ? const Color(0xFF101113) : Colors.grey.shade50,
      canvasColor: isDark ? const Color(0xFF1B1C1F) : Colors.white,
      cardColor: isDark ? const Color(0xFF1B1C1F) : Colors.white,
      dividerColor: isDark ? const Color(0xFF34363A) : const Color(0xFFE3E5E8),
      inputDecorationTheme: InputDecorationTheme(
        filled: true,
        fillColor: isDark ? const Color(0xFF222428) : Colors.white,
        border: OutlineInputBorder(
          borderRadius: BorderRadius.circular(5),
          borderSide: BorderSide(color: scheme.outlineVariant),
        ),
        enabledBorder: OutlineInputBorder(
          borderRadius: BorderRadius.circular(5),
          borderSide: BorderSide(color: scheme.outlineVariant),
        ),
      ),
      elevatedButtonTheme: ElevatedButtonThemeData(
        style: ElevatedButton.styleFrom(
          backgroundColor: isDark ? Colors.white : _brand,
          foregroundColor: isDark ? Colors.black : Colors.white,
        ),
      ),
      switchTheme: SwitchThemeData(
        thumbColor: WidgetStateProperty.resolveWith(
          (states) => states.contains(WidgetState.selected)
              ? (isDark ? Colors.black : Colors.white)
              : null,
        ),
        trackColor: WidgetStateProperty.resolveWith(
          (states) => states.contains(WidgetState.selected)
              ? (isDark ? Colors.white : _brand)
              : null,
        ),
      ),
    );
  }
}
