import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:psych_gen_app/core/designsystem/widgets/shimmer_image_placeholder.dart';
import 'package:psych_gen_app/core/utils/download_bytes.dart';
import 'package:psych_gen_app/features/face_generation/domain/entities/full_resolution_face.dart';

class QuickLookImageDialog extends StatefulWidget {
  final Uint8List previewBytes;
  final FullResolutionFace? fullResolutionFace;
  final Future<FullResolutionFace>? highResolutionFuture;
  final List<int> initialSelectedLevels;
  final List<int> levelCounts;
  final int horizontalAxis;
  final int? verticalAxis;
  final Uint8List? Function(List<int> levels) previewForLevels;
  final FullResolutionFace? Function(List<int> levels)
      cachedFullResolutionForLevels;
  final Future<FullResolutionFace> Function(
    List<int> levels,
    Uint8List previewBytes,
  ) requestFullResolutionForLevels;

  const QuickLookImageDialog({
    super.key,
    required this.previewBytes,
    this.fullResolutionFace,
    this.highResolutionFuture,
    required this.initialSelectedLevels,
    required this.levelCounts,
    this.horizontalAxis = 0,
    this.verticalAxis = 1,
    required this.previewForLevels,
    required this.cachedFullResolutionForLevels,
    required this.requestFullResolutionForLevels,
  });

  @override
  State<QuickLookImageDialog> createState() => _QuickLookImageDialogState();
}

class _QuickLookImageDialogState extends State<QuickLookImageDialog> {
  FullResolutionFace? _fullResolutionFace;
  Object? _error;
  late final String _downloadStem;
  late Uint8List _previewBytes;
  late List<int> _selectedLevels;
  int _selectionGeneration = 0;

  @override
  void initState() {
    super.initState();
    _fullResolutionFace = widget.fullResolutionFace;
    _previewBytes = widget.previewBytes;
    _selectedLevels = List<int>.from(widget.initialSelectedLevels);
    _downloadStem =
        'psychgen_${DateTime.now().millisecondsSinceEpoch.toString()}';
    _watchFullResolution(widget.highResolutionFuture, _selectionGeneration);
  }

  void _watchFullResolution(
    Future<FullResolutionFace>? future,
    int selectionGeneration,
  ) {
    future?.then((face) {
      if (mounted && selectionGeneration == _selectionGeneration) {
        setState(() => _fullResolutionFace = face);
      }
    }).catchError((Object error) {
      if (mounted && selectionGeneration == _selectionGeneration) {
        setState(() => _error = error);
      }
    });
  }

  KeyEventResult _handleKeyEvent(FocusNode node, KeyEvent event) {
    if (event is! KeyDownEvent && event is! KeyRepeatEvent) {
      return KeyEventResult.ignored;
    }
    if (event.logicalKey == LogicalKeyboardKey.arrowLeft) {
      return _navigate(axis: widget.horizontalAxis, delta: -1);
    }
    if (event.logicalKey == LogicalKeyboardKey.arrowRight) {
      return _navigate(axis: widget.horizontalAxis, delta: 1);
    }
    if (event.logicalKey == LogicalKeyboardKey.arrowUp) {
      return widget.verticalAxis == null
          ? KeyEventResult.handled
          : _navigate(axis: widget.verticalAxis!, delta: 1);
    }
    if (event.logicalKey == LogicalKeyboardKey.arrowDown) {
      return widget.verticalAxis == null
          ? KeyEventResult.handled
          : _navigate(axis: widget.verticalAxis!, delta: -1);
    }
    return KeyEventResult.ignored;
  }

  KeyEventResult _navigate({required int axis, required int delta}) {
    if (axis >= _selectedLevels.length || axis >= widget.levelCounts.length) {
      return KeyEventResult.handled;
    }
    final next = List<int>.from(_selectedLevels);
    final nextLevel = next[axis] + delta;
    if (nextLevel < 0 || nextLevel >= widget.levelCounts[axis]) {
      return KeyEventResult.handled;
    }
    next[axis] = nextLevel;
    final preview = widget.previewForLevels(next);
    if (preview == null) return KeyEventResult.handled;
    final cached = widget.cachedFullResolutionForLevels(next);
    final generation = ++_selectionGeneration;
    setState(() {
      _selectedLevels = next;
      _previewBytes = preview;
      _fullResolutionFace = cached;
      _error = null;
    });
    if (cached == null) {
      _watchFullResolution(
        widget.requestFullResolutionForLevels(next, preview),
        generation,
      );
    }
    return KeyEventResult.handled;
  }

  void _downloadImage() {
    final face = _fullResolutionFace;
    if (face == null) return;
    downloadBytes(
      face.imageBytes,
      filename: '${_downloadStem}_image.webp',
      mimeType: 'image/webp',
    );
  }

  void _downloadLatent() {
    final face = _fullResolutionFace;
    if (face == null) return;
    downloadBytes(
      face.latentNpyBytes,
      filename: '${_downloadStem}_latent.npy',
      mimeType: 'application/octet-stream',
    );
  }

  ButtonStyle _controlStyle() => IconButton.styleFrom(
        backgroundColor: Colors.black.withValues(alpha: 0.7),
        foregroundColor: Colors.white,
        disabledBackgroundColor: Colors.black.withValues(alpha: 0.35),
        disabledForegroundColor: Colors.white.withValues(alpha: 0.45),
      );

  @override
  Widget build(BuildContext context) {
    return Focus(
      autofocus: true,
      onKeyEvent: _handleKeyEvent,
      child: Material(
        color: Colors.transparent,
        child: Stack(
          children: [
            Positioned.fill(
              child: GestureDetector(
                behavior: HitTestBehavior.opaque,
                onTap: () => Navigator.of(context, rootNavigator: true).pop(),
              ),
            ),
            Positioned.fill(
              child: SafeArea(
                minimum: const EdgeInsets.all(28),
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    final size = constraints.maxWidth < constraints.maxHeight
                        ? constraints.maxWidth
                        : constraints.maxHeight;
                    return Center(
                      child: GestureDetector(
                        onTap: () {},
                        child: AsyncGeneratedImageTile(
                          imageBytes:
                              _fullResolutionFace?.imageBytes ?? _previewBytes,
                          size: size,
                          isLoading:
                              _fullResolutionFace == null && _error == null,
                          borderRadius: 10,
                        ),
                      ),
                    );
                  },
                ),
              ),
            ),
            if (_error != null)
              Positioned(
                left: 24,
                right: 24,
                bottom: 24,
                child: Center(
                  child: DecoratedBox(
                    decoration: BoxDecoration(
                      color: Colors.red.shade800,
                      borderRadius: BorderRadius.circular(8),
                    ),
                    child: const Padding(
                      padding:
                          EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                      child: Text(
                        'The full-resolution image could not be generated.',
                        style: TextStyle(color: Colors.white),
                      ),
                    ),
                  ),
                ),
              ),
            Positioned(
              right: 18,
              top: 18,
              child: SafeArea(
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Tooltip(
                      message: 'Download Image',
                      child: IconButton.filled(
                        style: _controlStyle(),
                        onPressed:
                            _fullResolutionFace == null ? null : _downloadImage,
                        icon: const Icon(Icons.download, size: 28),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Tooltip(
                      message: 'Download Latent',
                      child: IconButton.filled(
                        style: _controlStyle(),
                        onPressed: _fullResolutionFace == null
                            ? null
                            : _downloadLatent,
                        icon: const _LatentDownloadIcon(size: 28),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Tooltip(
                      message: 'Close quick look',
                      child: IconButton.filled(
                        style: _controlStyle(),
                        onPressed: () =>
                            Navigator.of(context, rootNavigator: true).pop(),
                        icon: const Icon(Icons.close, size: 28),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _LatentDownloadIcon extends StatelessWidget {
  final double size;

  const _LatentDownloadIcon({required this.size});

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: Size.square(size),
      painter: _LatentDownloadIconPainter(
        color: IconTheme.of(context).color ?? Colors.white,
      ),
    );
  }
}

class _LatentDownloadIconPainter extends CustomPainter {
  final Color color;

  const _LatentDownloadIconPainter({required this.color});

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..style = PaintingStyle.stroke
      ..strokeWidth = 2
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round;
    final trayTop = size.height * 0.72;
    final trayBottom = size.height * 0.88;
    final left = size.width * 0.18;
    final right = size.width * 0.82;
    final tray = Path()
      ..moveTo(left, trayTop)
      ..lineTo(left, trayBottom)
      ..lineTo(right, trayBottom)
      ..lineTo(right, trayTop);
    canvas.drawPath(tray, paint);
    final centerX = size.width * 0.5;
    final arrowTip = size.height * 0.68;
    canvas.drawLine(
      Offset(centerX, size.height * 0.40),
      Offset(centerX, arrowTip),
      paint,
    );
    canvas.drawLine(
      Offset(size.width * 0.36, size.height * 0.56),
      Offset(centerX, arrowTip),
      paint,
    );
    canvas.drawLine(
      Offset(size.width * 0.64, size.height * 0.56),
      Offset(centerX, arrowTip),
      paint,
    );

    final textPainter = TextPainter(
      text: TextSpan(
        text: '012...',
        style: TextStyle(
          color: color,
          fontSize: size.width * 0.23,
          fontWeight: FontWeight.w700,
          height: 1,
        ),
      ),
      textDirection: TextDirection.ltr,
      maxLines: 1,
    )..layout(maxWidth: size.width);
    textPainter.paint(
      canvas,
      Offset((size.width - textPainter.width) / 2, size.height * 0.04),
    );
  }

  @override
  bool shouldRepaint(covariant _LatentDownloadIconPainter oldDelegate) =>
      oldDelegate.color != color;
}
