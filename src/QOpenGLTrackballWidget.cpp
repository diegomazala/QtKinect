
#include "QOpenGLTrackballWidget.h"
#include <QMouseEvent>



QOpenGLTrackballWidget::QOpenGLTrackballWidget(QWidget *parent) :
	QOpenGLWidget(parent),
	angularSpeed(0),
	fovy(60.0f),
	nearPlane(0.1f),
	farPlane(1024.f)
{
	timer.start(12, this);
}

QOpenGLTrackballWidget::~QOpenGLTrackballWidget()
{
}

void QOpenGLTrackballWidget::mousePressEvent(QMouseEvent *e)
{
	// Save mouse press position
	mousePressPosition = QVector2D(e->localPos());
	angularSpeed = 0.0;
}


void QOpenGLTrackballWidget::mouseMoveEvent(QMouseEvent *e)
{
	// Mouse release position - mouse press position
	QVector2D diff = QVector2D(e->localPos()) - mousePressPosition;

	// Rotation axis is perpendicular to the mouse position difference
	// vector
	QVector3D n = QVector3D(diff.y(), diff.x(), 0.0).normalized();

	// Accelerate angular speed relative to the length of the mouse sweep
	qreal acc = diff.length() / 100.0;

	// Calculate new rotation axis as weighted sum
	//rotationAxis = (rotationAxis * angularSpeed + n * acc).normalized();
	rotationAxis = n.normalized() * acc;

	// Increase angular speed
	//angularSpeed += acc;
	angularSpeed = acc * 10.0;

	mousePressPosition = QVector2D(e->localPos());
}


void QOpenGLTrackballWidget::mouseReleaseEvent(QMouseEvent *e)
{
	e->accept();
}


void QOpenGLTrackballWidget::wheelEvent(QWheelEvent* event)
{
	position.setZ( position.z() - event->delta() * weelSpeed);

	//if (distance < 0.5f)
	//	distance = 0.5f;

	//if (distance > 10240.f)
	//	distance = 10240.f;

	event->accept();

	update();
}


void QOpenGLTrackballWidget::timerEvent(QTimerEvent *)
{
	// Decrease angular speed (friction)
	angularSpeed *= 0.95;

	// Stop rotation when speed goes below threshold
	if (angularSpeed < 0.01) 
	{
		angularSpeed = 0.0;
	}
	else 
	{
		// Update rotation
		rotation = QQuaternion::fromAxisAndAngle(rotationAxis, angularSpeed) * rotation;

		// Request an update
		update();
	}
}


void QOpenGLTrackballWidget::keyReleaseEvent(QKeyEvent *e)
{
	if (e->key() == Qt::Key_Q || e->key() == Qt::Key_Escape)
		this->close();
}



void QOpenGLTrackballWidget::resizeGL(int w, int h)
{
	projection.setToIdentity();	// Reset projection
	projection.perspective(
		fovy, 
		(float)w / (float)h, 
		nearPlane, 
		farPlane);	// Set perspective projection
}
