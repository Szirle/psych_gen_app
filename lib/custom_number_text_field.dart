import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

class CustomNumberTextField extends StatefulWidget {
  final TextEditingController? controller;
  final FocusNode? focusNode;
  final int min;
  final int max;
  final int step;
  final double arrowsWidth;
  final double arrowsHeight;
  final EdgeInsets contentPadding;
  final double borderWidth;
  final ValueChanged<int?>? onChanged;

  const CustomNumberTextField({
    Key? key,
    this.controller,
    this.focusNode,
    this.min = 0,
    this.max = 99999,
    this.step = 1,
    this.arrowsWidth = 24,
    this.arrowsHeight = kMinInteractiveDimension,
    this.contentPadding = const EdgeInsets.symmetric(horizontal: 8),
    this.borderWidth = 2,
    this.onChanged,
  }) : super(key: key);

  @override
  State<StatefulWidget> createState() => _CustomNumberTextFieldState();
}

class _CustomNumberTextFieldState extends State<CustomNumberTextField> {
  late TextEditingController _controller;
  late FocusNode _focusNode;
  bool _canGoUp = false;
  bool _canGoDown = false;

  @override
  void initState() {
    super.initState();
    _controller = widget.controller ?? TextEditingController();
    _focusNode = widget.focusNode ?? FocusNode();
    _updateArrows(int.tryParse(_controller.text));
  }

  @override
  void didUpdateWidget(covariant CustomNumberTextField oldWidget) {
    super.didUpdateWidget(oldWidget);
    _controller = widget.controller ?? _controller;
    _focusNode = widget.focusNode ?? _focusNode;
    _updateArrows(int.tryParse(_controller.text));
  }

  @override
  Widget build(BuildContext context) => SizedBox(
      height: 36, child: TextField(
      controller: _controller,
      focusNode: _focusNode,
      textInputAction: TextInputAction.done,
      keyboardType: TextInputType.number,
      maxLength: widget.max.toString().length + (widget.min.isNegative ? 1 : 0),
      decoration: InputDecoration(
          fillColor: Colors.white,
          focusedBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(5.0),
            borderSide: const BorderSide(color: Colors.black26, width: 1.0),
          ),
          enabledBorder: OutlineInputBorder(
            borderRadius: BorderRadius.circular(5.0),
            borderSide: const BorderSide(color: Colors.black26, width: 1.0),
          ),
          contentPadding: const EdgeInsets.only(top: 12, left: 12, right: 12),
          counterText: '',
          isDense: true,
          filled: true,
          suffixIconConstraints: BoxConstraints(
              maxHeight: widget.arrowsHeight,
              maxWidth: widget.arrowsWidth + widget.contentPadding.right),
          suffixIcon: Container(
              decoration: BoxDecoration(
                  borderRadius: BorderRadius.only(
                      topRight: Radius.circular(widget.borderWidth),
                      bottomRight: Radius.circular(widget.borderWidth))),
              clipBehavior: Clip.antiAlias,
              alignment: Alignment.centerRight,
              margin: EdgeInsets.only(
                  top: widget.borderWidth,
                  right: widget.borderWidth,
                  bottom: widget.borderWidth,
                  left: widget.contentPadding.right),
              child: Column(crossAxisAlignment: CrossAxisAlignment.stretch, children: [
                Expanded(
                    child: Material(
                        type: MaterialType.transparency,
                        child: InkWell(
                            child: Opacity(
                                opacity: _canGoUp ? 1 : .5, child: const Icon(Icons.arrow_drop_up)),
                            onTap: _canGoUp ? () => _update(true) : null))),
                Expanded(
                    child: Material(
                        type: MaterialType.transparency,
                        child: InkWell(
                            child: Opacity(
                                opacity: _canGoDown ? 1 : .5,
                                child: const Icon(Icons.arrow_drop_down)),
                            onTap: _canGoDown ? () => _update(false) : null))),
              ]))),
      maxLines: 1,
      onChanged: (value) {
        final intValue = int.tryParse(value);
        widget.onChanged?.call(intValue);
        _updateArrows(intValue);
      }));

  void _update(bool up) {
    var intValue = int.tryParse(_controller.text);
    intValue == null ? intValue = 0 : intValue += up ? widget.step : -widget.step;
    _controller.text = intValue.toString();
    _updateArrows(intValue);
    _focusNode.requestFocus();
  }

  void _updateArrows(int? value) {
    final canGoUp = value == null || value < widget.max;
    final canGoDown = value == null || value > widget.min;
    if (_canGoUp != canGoUp || _canGoDown != canGoDown)
      setState(() {
        _canGoUp = canGoUp;
        _canGoDown = canGoDown;
      });
  }
}

