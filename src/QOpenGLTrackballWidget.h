
#ifndef __Q_OPENGL_TRACKBALL_WIDGET_H__
#define __Q_OPENGL_TRACKBALL_WIDGET_H__

#include <QOpenGLWidget>
#include <QQuaternion>
#include <QVector2D>
#include <QBasicTimer>

class QOpenGLTrackballWidget : public QOpenGLWidget
{
public:
	explicit QOpenGLTrackballWidget(QWidget *parent = 0);
	~QOpenGLTrackballWidget();

protected:
	void mousePressEvent(QMouseEvent *e) Q_DECL_OVERRIDE;
	void mouseMoveEvent(QMouseEvent *e) Q_DECL_OVERRIDE;
	void mouseReleaseEvent(QMouseEvent *e) Q_DECL_OVERRIDE;
	void wheelEvent(QWheelEvent* e) Q_DECL_OVERRIDE;
	void timerEvent(QTimerEvent *e) Q_DECL_OVERRIDE;

	QBasicTimer timer;
	QVector2D mousePressPosition;
	QVector3D rotationAxis;
	qreal angularSpeed;
	QQuaternion rotation;
	float distance;
};



#endif // __Q_OPENGL_TRACKBALL_WIDGET_H__
